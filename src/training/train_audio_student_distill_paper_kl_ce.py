import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import matplotlib
matplotlib.use("Agg")
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf

from audio_student_model import build_audio_student_model, build_optimizer
from train_audio_student_distill import (
    load_config,
    plot_history,
    save_confusion_matrix_figure,
    select_labels,
)
from train_audio_student_distill_tcn_teacher import (
    aggregate_teacher_probabilities,
    build_audio_student,
    build_grouped_dataset_for_tcn_teacher,
    build_tcn_teacher_model,
    class_names_from_map,
    evaluate_teacher,
    flatten_teacher_groups,
    load_tcn_teacher_model,
)


PAPER_SETTINGS = {
    "temperature": 3.0,
    "alpha": 0.4,
}

DEFAULT_TIME_ALIGNED_TEACHER = Path(
    "results/original_time_aligned_tcn_teacher_outputs/teacher_vibration.keras"
)
DEFAULT_TIME_ALIGNED_AUDIO_STUDENT = Path(
    "results/original_time_aligned_audio_spectrogram_outputs/audio_spectrogram_cnn.keras"
)


class PaperDistillationTrainer(keras.Model):
    def __init__(
        self,
        teacher,
        student,
        alpha,
        temperature,
        confidence_mode,
        confidence_threshold,
        ce_class_weights=None,
        focus_label_index=None,
        focus_kl_threshold=0.0,
        focus_kl_boost=1.0,
        focus_margin_label_index=None,
        focus_margin_rival_index=None,
        focus_margin_value=0.0,
        focus_margin_weight=0.0,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.confidence_mode = str(confidence_mode)
        self.confidence_threshold = float(confidence_threshold)
        self.ce_class_weights = None if ce_class_weights is None else tf.constant(
            np.asarray(ce_class_weights, dtype=np.float32)
        )
        self.focus_label_index = None if focus_label_index is None else int(focus_label_index)
        self.focus_kl_threshold = float(focus_kl_threshold)
        self.focus_kl_boost = float(focus_kl_boost)
        self.focus_margin_label_index = (
            None if focus_margin_label_index is None else int(focus_margin_label_index)
        )
        self.focus_margin_rival_index = (
            None if focus_margin_rival_index is None else int(focus_margin_rival_index)
        )
        self.focus_margin_value = float(focus_margin_value)
        self.focus_margin_weight = float(focus_margin_weight)
        self.student_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.ce_tracker = keras.metrics.Mean(name="ce_loss")
        self.kl_tracker = keras.metrics.Mean(name="kl_loss")
        self.margin_tracker = keras.metrics.Mean(name="margin_loss")
        self.confidence_tracker = keras.metrics.Mean(name="confidence_weight")
        self.acc_metric = keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.teacher.trainable = False

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.ce_tracker,
            self.kl_tracker,
            self.margin_tracker,
            self.confidence_tracker,
            self.acc_metric,
        ]

    def _compute_losses(self, audio_batch, teacher_soft_batch, labels, training):
        student_logits = self.student(audio_batch, training=training)
        ce_per_sample = keras.losses.sparse_categorical_crossentropy(
            labels, student_logits, from_logits=True
        )
        if self.ce_class_weights is not None:
            ce_weights = tf.gather(self.ce_class_weights, tf.cast(labels, tf.int32))
            ce_per_sample = ce_per_sample * ce_weights
        student_soft = tf.nn.softmax(student_logits / self.temperature, axis=-1)
        kl_per_sample = tf.reduce_sum(
            teacher_soft_batch
            * (
                tf.math.log(tf.clip_by_value(teacher_soft_batch, 1e-8, 1.0))
                - tf.math.log(tf.clip_by_value(student_soft, 1e-8, 1.0))
            ),
            axis=-1,
        )

        teacher_confidence = tf.reduce_max(teacher_soft_batch, axis=-1)
        if self.confidence_mode == "soft":
            kl_weight = teacher_confidence
        elif self.confidence_mode == "hard":
            kl_weight = tf.cast(
                teacher_confidence >= self.confidence_threshold, tf.float32
            )
        else:
            kl_weight = tf.ones_like(teacher_confidence, dtype=tf.float32)

        if self.focus_label_index is not None and self.focus_kl_boost > 1.0:
            focus_mask = tf.equal(tf.cast(labels, tf.int32), self.focus_label_index)
            if self.focus_kl_threshold > 0.0:
                focus_mask = tf.logical_and(
                    focus_mask,
                    teacher_confidence >= self.focus_kl_threshold,
                )
            focus_boost = tf.where(
                focus_mask,
                tf.fill(tf.shape(teacher_confidence), tf.cast(self.focus_kl_boost, tf.float32)),
                tf.ones_like(teacher_confidence, dtype=tf.float32),
            )
            kl_weight = kl_weight * focus_boost

        margin_loss = tf.constant(0.0, dtype=tf.float32)
        if (
            self.focus_margin_label_index is not None
            and self.focus_margin_rival_index is not None
            and self.focus_margin_weight > 0.0
            and self.focus_margin_value > 0.0
        ):
            focus_logits = student_logits[:, self.focus_margin_label_index]
            rival_logits = student_logits[:, self.focus_margin_rival_index]
            focus_margin_mask = tf.equal(
                tf.cast(labels, tf.int32), self.focus_margin_label_index
            )
            per_sample_margin = tf.nn.relu(
                self.focus_margin_value - (focus_logits - rival_logits)
            )
            masked_margin = tf.where(
                focus_margin_mask,
                per_sample_margin,
                tf.zeros_like(per_sample_margin),
            )
            margin_loss = tf.math.divide_no_nan(
                tf.reduce_sum(masked_margin),
                tf.reduce_sum(tf.cast(focus_margin_mask, tf.float32)),
            )

        total_per_sample = (1.0 - self.alpha) * ce_per_sample + self.alpha * (
            kl_weight * kl_per_sample
        )
        total_loss = tf.reduce_mean(total_per_sample) + self.focus_margin_weight * margin_loss
        ce_loss = tf.reduce_mean(ce_per_sample)
        kl_loss = tf.math.divide_no_nan(
            tf.reduce_sum(kl_weight * kl_per_sample),
            tf.reduce_sum(kl_weight),
        )
        confidence_weight = tf.reduce_mean(kl_weight)
        return total_loss, ce_loss, kl_loss, margin_loss, confidence_weight, student_logits

    def train_step(self, data):
        (audio_batch, teacher_soft_batch), labels = data
        with tf.GradientTape() as tape:
            total_loss, ce_loss, kl_loss, margin_loss, confidence_weight, student_logits = self._compute_losses(
                audio_batch, teacher_soft_batch, labels, training=True
            )
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        self.loss_tracker.update_state(total_loss)
        self.ce_tracker.update_state(ce_loss)
        self.kl_tracker.update_state(kl_loss)
        self.margin_tracker.update_state(margin_loss)
        self.confidence_tracker.update_state(confidence_weight)
        self.acc_metric.update_state(labels, student_logits)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        (audio_batch, teacher_soft_batch), labels = data
        total_loss, ce_loss, kl_loss, margin_loss, confidence_weight, student_logits = self._compute_losses(
            audio_batch, teacher_soft_batch, labels, training=False
        )
        self.loss_tracker.update_state(total_loss)
        self.ce_tracker.update_state(ce_loss)
        self.kl_tracker.update_state(kl_loss)
        self.margin_tracker.update_state(margin_loss)
        self.confidence_tracker.update_state(confidence_weight)
        self.acc_metric.update_state(labels, student_logits)
        return {metric.name: metric.result() for metric in self.metrics}


def teacher_soft_labels_from_outputs(teacher_outputs, temperature):
    teacher_outputs = np.asarray(teacher_outputs, dtype=np.float32)
    if teacher_outputs.size == 0:
        return teacher_outputs

    row_sums = np.sum(teacher_outputs, axis=-1)
    looks_like_probs = (
        np.all(teacher_outputs >= 0.0)
        and np.all(teacher_outputs <= 1.0)
        and np.allclose(row_sums, 1.0, atol=1e-3)
    )
    if looks_like_probs:
        clipped = np.clip(teacher_outputs, 1e-8, 1.0)
        return tf.nn.softmax(np.log(clipped) / float(temperature), axis=-1).numpy()

    return tf.nn.softmax(teacher_outputs / float(temperature), axis=-1).numpy()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Paper-style vibration-to-audio distillation with KL + CE. "
            "Teacher uses vibration windows, student uses aligned audio spectrograms."
        )
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--audio-target-sr", type=int, default=16000)
    parser.add_argument("--event-duration-sec", type=float, default=0.8)
    parser.add_argument(
        "--hop-sec",
        type=float,
        default=0.8,
        help="Window hop in seconds. Default matches the window size to minimize overlap.",
    )
    parser.add_argument("--audio-highpass-hz", type=float, default=80.0)
    parser.add_argument("--nperseg", type=int, default=512)
    parser.add_argument("--noverlap", type=int, default=384)
    parser.add_argument("--max-freq-hz", type=float, default=6000.0)
    parser.add_argument("--teacher-window-size", type=int, default=243)
    parser.add_argument("--audio-offset-sec", type=float, default=0.0)
    parser.add_argument("--vibration-offset-sec", type=float, default=0.0)
    parser.add_argument("--shared-duration-sec", type=float, default=None)
    parser.add_argument("--teacher-epochs", type=int, default=60)
    parser.add_argument("--student-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--alpha",
        type=float,
        default=PAPER_SETTINGS["alpha"],
        help="Paper-style balance coefficient λ for KL + CE.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=PAPER_SETTINGS["temperature"],
        help="Paper-style temperature T for teacher soft labels.",
    )
    parser.add_argument(
        "--confidence-mode",
        choices=["none", "soft", "hard"],
        default="soft",
        help="How teacher confidence scales KL. 'soft' uses max teacher prob, 'hard' gates by threshold.",
    )
    parser.add_argument(
        "--teacher-confidence-threshold",
        type=float,
        default=0.75,
        help="Threshold used when confidence-mode=hard.",
    )
    parser.add_argument(
        "--student-class-weight",
        nargs="*",
        default=None,
        help="Optional CE class weights as label=value pairs, e.g. bottle_can_330=1.5.",
    )
    parser.add_argument(
        "--focus-kl-label",
        type=str,
        default=None,
        help="Optional label name whose high-confidence samples receive boosted KL weight.",
    )
    parser.add_argument(
        "--focus-kl-threshold",
        type=float,
        default=0.0,
        help="Teacher confidence threshold required before boosting KL on the focused label.",
    )
    parser.add_argument(
        "--focus-kl-boost",
        type=float,
        default=1.0,
        help="Multiplier applied to KL weight for focused-label samples when the threshold is met.",
    )
    parser.add_argument(
        "--focus-margin-label",
        type=str,
        default=None,
        help="Optional label name whose samples receive an extra rival-separation margin loss.",
    )
    parser.add_argument(
        "--focus-margin-rival-label",
        type=str,
        default=None,
        help="Optional rival label used by the extra margin loss.",
    )
    parser.add_argument(
        "--focus-margin-value",
        type=float,
        default=0.0,
        help="Desired minimum logit gap between focus-margin-label and rival label.",
    )
    parser.add_argument(
        "--focus-margin-weight",
        type=float,
        default=0.0,
        help="Weight of the focus-label rival-separation margin loss.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--student-init-model",
        type=Path,
        default=DEFAULT_TIME_ALIGNED_AUDIO_STUDENT,
        help=(
            "Initial audio student checkpoint. Defaults to the time-aligned audio "
            "spectrogram model. If the file is missing, the student is trained from scratch."
        ),
    )
    parser.add_argument(
        "--split-mode",
        choices=["blocked_time", "random"],
        default="blocked_time",
        help="Use blocked_time to avoid temporal leakage across windows from the same recording.",
    )
    parser.add_argument(
        "--teacher-model",
        type=Path,
        default=DEFAULT_TIME_ALIGNED_TEACHER,
        help=(
            "Pre-trained vibration teacher. Defaults to the time-aligned TCN teacher "
            "saved by train_time_aligned_tcn_teacher.py. If the file is missing, "
            "this script falls back to training a teacher in-script."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/distill_outputs_paper_kl_ce_old"),
    )
    return parser.parse_args()


def split_count_triplet(count):
    train_count = int(round(count * 0.7))
    val_count = int(round(count * 0.15))
    test_count = count - train_count - val_count

    if count >= 3:
        if train_count <= 0:
            train_count = 1
        if val_count <= 0:
            val_count = 1
        test_count = count - train_count - val_count
        if test_count <= 0:
            test_count = 1
            if train_count >= val_count and train_count > 1:
                train_count -= 1
            elif val_count > 1:
                val_count -= 1
    return train_count, val_count, test_count


def parse_label_weight_overrides(entries, label_to_int):
    if not entries:
        return None
    weights = np.ones((len(label_to_int),), dtype=np.float32)
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --student-class-weight entry '{entry}'. Use label=value."
            )
        label, raw_value = entry.split("=", 1)
        label = label.strip()
        if label not in label_to_int:
            known = ", ".join(sorted(label_to_int))
            raise ValueError(
                f"Unknown label '{label}' in --student-class-weight. Known labels: {known}."
            )
        weights[label_to_int[label]] = float(raw_value)
    return weights


def split_grouped_dataset(x_audio, teacher_window_groups, y, dataset_stats, args):
    if args.split_mode == "random":
        split_payload = train_test_split(
            x_audio,
            np.arange(len(y), dtype=np.int32),
            y,
            test_size=0.3,
            random_state=args.random_state,
            stratify=y,
        )
        (
            x_audio_train,
            x_audio_temp,
            group_index_train,
            group_index_temp,
            y_train,
            y_temp,
        ) = split_payload

        split_payload = train_test_split(
            x_audio_temp,
            group_index_temp,
            y_temp,
            test_size=0.5,
            random_state=args.random_state,
            stratify=y_temp,
        )
        (
            x_audio_val,
            x_audio_test,
            group_index_val,
            group_index_test,
            y_val,
            y_test,
        ) = split_payload

        teacher_groups_train = [teacher_window_groups[index] for index in group_index_train]
        teacher_groups_val = [teacher_window_groups[index] for index in group_index_val]
        teacher_groups_test = [teacher_window_groups[index] for index in group_index_test]
        return (
            x_audio_train,
            x_audio_val,
            x_audio_test,
            teacher_groups_train,
            teacher_groups_val,
            teacher_groups_test,
            y_train,
            y_val,
            y_test,
        )

    x_train_batches = []
    x_val_batches = []
    x_test_batches = []
    groups_train = []
    groups_val = []
    groups_test = []
    y_train_batches = []
    y_val_batches = []
    y_test_batches = []

    start = 0
    for item in dataset_stats:
        count = int(item["paired_window_count"])
        end = start + count
        x_label = x_audio[start:end]
        y_label = y[start:end]
        group_label = teacher_window_groups[start:end]
        start = end

        train_count, val_count, test_count = split_count_triplet(count)
        train_end = train_count
        val_end = train_count + val_count

        x_train_batches.append(x_label[:train_end])
        x_val_batches.append(x_label[train_end:val_end])
        x_test_batches.append(x_label[val_end:val_end + test_count])

        groups_train.extend(group_label[:train_end])
        groups_val.extend(group_label[train_end:val_end])
        groups_test.extend(group_label[val_end:val_end + test_count])

        y_train_batches.append(y_label[:train_end])
        y_val_batches.append(y_label[train_end:val_end])
        y_test_batches.append(y_label[val_end:val_end + test_count])

    return (
        np.concatenate(x_train_batches, axis=0),
        np.concatenate(x_val_batches, axis=0),
        np.concatenate(x_test_batches, axis=0),
        groups_train,
        groups_val,
        groups_test,
        np.concatenate(y_train_batches, axis=0),
        np.concatenate(y_val_batches, axis=0),
        np.concatenate(y_test_batches, axis=0),
    )


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.random_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paired_config = select_labels(load_config(args.config), args.labels)
    teacher_window_size = args.teacher_window_size

    teacher_model_path = args.teacher_model
    if teacher_model_path is not None and not Path(teacher_model_path).exists():
        print(
            f"Teacher model not found at {teacher_model_path}. "
            "Falling back to training a teacher inside this script."
        )
        teacher_model_path = None

    if teacher_model_path is not None:
        teacher_probe = load_tcn_teacher_model(teacher_model_path)
        teacher_window_size = int(teacher_probe.input_shape[1])

    x_audio, teacher_window_groups, y, label_to_int, dataset_stats = build_grouped_dataset_for_tcn_teacher(
        paired_config,
        args,
        teacher_window_size=teacher_window_size,
    )

    (
        x_audio_train,
        x_audio_val,
        x_audio_test,
        teacher_groups_train,
        teacher_groups_val,
        teacher_groups_test,
        y_train,
        y_val,
        y_test,
    ) = split_grouped_dataset(
        x_audio,
        teacher_window_groups,
        y,
        dataset_stats,
        args,
    )

    ce_class_weights = parse_label_weight_overrides(args.student_class_weight, label_to_int)
    focus_label_index = None
    if args.focus_kl_label is not None:
        if args.focus_kl_label not in label_to_int:
            known = ", ".join(sorted(label_to_int))
            raise ValueError(
                f"Unknown --focus-kl-label '{args.focus_kl_label}'. Known labels: {known}."
            )
        focus_label_index = label_to_int[args.focus_kl_label]

    focus_margin_label_index = None
    focus_margin_rival_index = None
    if args.focus_margin_label is not None or args.focus_margin_rival_label is not None:
        if not args.focus_margin_label or not args.focus_margin_rival_label:
            raise ValueError(
                "--focus-margin-label and --focus-margin-rival-label must be provided together."
            )
        if args.focus_margin_label not in label_to_int:
            known = ", ".join(sorted(label_to_int))
            raise ValueError(
                f"Unknown --focus-margin-label '{args.focus_margin_label}'. Known labels: {known}."
            )
        if args.focus_margin_rival_label not in label_to_int:
            known = ", ".join(sorted(label_to_int))
            raise ValueError(
                f"Unknown --focus-margin-rival-label '{args.focus_margin_rival_label}'. Known labels: {known}."
            )
        focus_margin_label_index = label_to_int[args.focus_margin_label]
        focus_margin_rival_index = label_to_int[args.focus_margin_rival_label]

    if teacher_model_path is not None:
        teacher = load_tcn_teacher_model(teacher_model_path)
        if int(teacher.output_shape[-1]) != int(len(label_to_int)):
            raise ValueError(
                f"Teacher model outputs {teacher.output_shape[-1]} classes, but this run expects {len(label_to_int)}."
            )
        teacher.compile(
            optimizer=build_optimizer(),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
    else:
        x_vibration_train, y_vibration_train = flatten_teacher_groups(teacher_groups_train, y_train)
        x_vibration_val, y_vibration_val = flatten_teacher_groups(teacher_groups_val, y_val)
        teacher = build_tcn_teacher_model(x_vibration_train.shape[1:], len(label_to_int))
        teacher.compile(
            optimizer=build_optimizer(),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        teacher_history = teacher.fit(
            x_vibration_train,
            y_vibration_train,
            validation_data=(x_vibration_val, y_vibration_val),
            epochs=args.teacher_epochs,
            batch_size=args.batch_size,
            verbose=2,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=100,
                    restore_best_weights=True,
                )
            ],
        )
        plot_history(
            teacher_history,
            args.output_dir / "teacher_accuracy_loss.png",
            "Paper-Style Vibration Teacher Training Curves",
            ["accuracy", "loss"],
        )
        teacher.save(args.output_dir / "teacher_tcn.keras")

    teacher_metrics, teacher_report = evaluate_teacher(
        teacher,
        teacher_groups_test,
        y_test,
        label_to_int,
        args.output_dir,
        batch_size=args.batch_size,
    )

    teacher_soft_train = teacher_soft_labels_from_outputs(
        aggregate_teacher_probabilities(teacher, teacher_groups_train, batch_size=args.batch_size),
        args.temperature,
    )
    teacher_soft_val = teacher_soft_labels_from_outputs(
        aggregate_teacher_probabilities(teacher, teacher_groups_val, batch_size=args.batch_size),
        args.temperature,
    )

    student = load_or_build_student(args, x_audio_train.shape[1:], len(label_to_int))
    distiller = PaperDistillationTrainer(
        teacher=teacher,
        student=student,
        alpha=args.alpha,
        temperature=args.temperature,
        confidence_mode=args.confidence_mode,
        confidence_threshold=args.teacher_confidence_threshold,
        ce_class_weights=ce_class_weights,
        focus_label_index=focus_label_index,
        focus_kl_threshold=args.focus_kl_threshold,
        focus_kl_boost=args.focus_kl_boost,
        focus_margin_label_index=focus_margin_label_index,
        focus_margin_rival_index=focus_margin_rival_index,
        focus_margin_value=args.focus_margin_value,
        focus_margin_weight=args.focus_margin_weight,
    )
    distiller.compile(optimizer=build_optimizer())

    train_ds = tf_dataset_from_soft_labels(
        x_audio_train, teacher_soft_train, y_train, args.batch_size, training=True
    )
    val_ds = tf_dataset_from_soft_labels(
        x_audio_val, teacher_soft_val, y_val, args.batch_size, training=False
    )
    student_history = distiller.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.student_epochs,
        verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=100,
                restore_best_weights=True,
            )
        ],
    )

    plot_history(
        student_history,
        args.output_dir / "student_accuracy_loss.png",
        (
            "Paper-Style Student Distillation Curves "
            f"(lambda={args.alpha:.2f}, T={args.temperature:.1f})"
        ),
        ["accuracy", "loss"],
    )
    plot_history(
        student_history,
        args.output_dir / "student_ce_kl.png",
        (
            "Paper-Style Student CE and KL "
            f"(lambda={args.alpha:.2f}, T={args.temperature:.1f})"
        ),
        ["ce_loss", "kl_loss", "confidence_weight"],
    )

    student.compile(
        optimizer=build_optimizer(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    student_metrics = student.evaluate(
        x_audio_test, y_test, batch_size=args.batch_size, verbose=0, return_dict=True
    )
    class_names = class_names_from_map(label_to_int)
    student_logits = student.predict(x_audio_test, batch_size=args.batch_size, verbose=0)
    student_pred = np.argmax(student_logits, axis=1)
    student_report = classification_report(
        y_test,
        student_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    save_confusion_matrix_figure(
        y_test,
        student_pred,
        class_names,
        args.output_dir / "student_confusion_matrix.png",
    )
    with (args.output_dir / "student_classification_report.txt").open("w", encoding="utf-8") as handle:
        handle.write(student_report)
    student.save(args.output_dir / "student_audio_only.keras")

    with (args.output_dir / "label_to_int.json").open("w", encoding="utf-8") as handle:
        json.dump(label_to_int, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "paired_dataset_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_stats, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "distill_run_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "paper_reference": "electronics-15-01631-v2.pdf",
                "distillation_loss": "L = (1-lambda) * CE + lambda * KL",
                "paper_temperature": 3.0,
                "paper_lambda": 0.7,
                "dataset_protocol": "default_window_split",
                "teacher_model": str(teacher_model_path) if teacher_model_path is not None else "trained_in_script",
                "teacher_window_size": int(teacher_window_size),
                "alpha": args.alpha,
                "temperature": args.temperature,
                "confidence_mode": args.confidence_mode,
                "teacher_confidence_threshold": args.teacher_confidence_threshold,
                "student_class_weight": (
                    None
                    if ce_class_weights is None
                    else {
                        label: float(ce_class_weights[index])
                        for label, index in sorted(label_to_int.items(), key=lambda item: item[1])
                    }
                ),
                "focus_kl_label": args.focus_kl_label,
                "focus_kl_threshold": args.focus_kl_threshold,
                "focus_kl_boost": args.focus_kl_boost,
                "focus_margin_label": args.focus_margin_label,
                "focus_margin_rival_label": args.focus_margin_rival_label,
                "focus_margin_value": args.focus_margin_value,
                "focus_margin_weight": args.focus_margin_weight,
                "student_init_model": (
                    str(args.student_init_model) if args.student_init_model is not None else None
                ),
                "split_mode": args.split_mode,
                "audio_target_sr": args.audio_target_sr,
                "audio_offset_sec": args.audio_offset_sec,
                "vibration_offset_sec": args.vibration_offset_sec,
                "shared_duration_sec": args.shared_duration_sec,
                "event_duration_sec": args.event_duration_sec,
                "hop_sec": args.hop_sec,
                "audio_highpass_hz": args.audio_highpass_hz,
                "nperseg": args.nperseg,
                "noverlap": args.noverlap,
                "max_freq_hz": args.max_freq_hz,
                "teacher_test_loss": float(teacher_metrics["loss"]),
                "teacher_test_accuracy": float(teacher_metrics["accuracy"]),
                "student_test_loss": float(student_metrics["loss"]),
                "student_test_accuracy": float(student_metrics["accuracy"]),
                "label_to_int": label_to_int,
                "student_input_shape": list(x_audio_train.shape[1:]),
                "student_input_type": "log_spectrogram",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("Paper-style vibration teacher classification report:")
    print(teacher_report)
    print("Paper-style audio student classification report:")
    print(student_report)


def load_or_build_student(args, input_shape, num_classes):
    init_path = args.student_init_model
    if init_path is None or not Path(init_path).exists():
        if init_path is not None and not Path(init_path).exists():
            print(
                f"Student init model not found at {init_path}. "
                "Falling back to training the student from scratch."
            )
        return build_audio_student(input_shape, num_classes)

    student = keras.models.load_model(init_path, compile=False)
    expected_input_shape = tuple(input_shape)
    actual_input_shape = tuple(student.input_shape[1:])
    if actual_input_shape != expected_input_shape:
        raise ValueError(
            f"Student init model input shape {actual_input_shape} does not match expected {expected_input_shape}."
        )
    output_units = int(student.output_shape[-1])
    if output_units != int(num_classes):
        raise ValueError(
            f"Student init model outputs {output_units} classes, but this run expects {num_classes}."
        )
    return student


def tf_dataset_from_soft_labels(audio_data, teacher_soft_data, labels, batch_size, training):
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices(((audio_data, teacher_soft_data), labels))
    if training:
        dataset = dataset.shuffle(len(labels), reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    main()
