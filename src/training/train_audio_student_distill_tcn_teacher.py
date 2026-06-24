import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import matplotlib
matplotlib.use("Agg")
import numpy as np
from tensorflow import keras

from audio_student_model import (
    build_audio_student_model,
    build_optimizer,
    compute_log_spectrograms,
    load_audio_signal,
    sliding_windows_1d,
)
from event_windows import (
    crop_vibration_to_time_range,
    load_vibration_frame,
    resample_vibration_interval,
    synchronized_time_range,
    zscore_normalize,
)
from layers.tcn import TCN
from train_audio_student_distill import (
    load_config,
    plot_history,
    save_confusion_matrix_figure,
    select_labels,
)
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parent
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "No-confidence vibration-to-audio distillation with KL + CE. "
            "Teacher uses aligned vibration windows, student uses aligned audio spectrograms."
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
    parser.add_argument(
        "--audio-offset-sec",
        type=float,
        default=0.0,
        help="Global audio start offset in seconds. Positive values trim the audio front before alignment.",
    )
    parser.add_argument(
        "--vibration-offset-sec",
        type=float,
        default=0.0,
        help="Global vibration start offset in seconds. Positive values trim the vibration front before alignment.",
    )
    parser.add_argument(
        "--shared-duration-sec",
        type=float,
        default=None,
        help="Optional manual aligned duration in seconds. By default the shared overlap is used.",
    )
    parser.add_argument("--teacher-epochs", type=int, default=60)
    parser.add_argument("--student-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--alpha",
        type=float,
        default=PAPER_SETTINGS["alpha"],
        help="Balance coefficient lambda for KL + CE.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=PAPER_SETTINGS["temperature"],
        help="Temperature T for teacher soft labels.",
    )
    parser.add_argument(
        "--student-class-weight",
        nargs="*",
        default=None,
        help="Optional CE class weights as label=value pairs, e.g. bottle_can_330=1.5.",
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
        help="Pre-trained vibration teacher checkpoint used for the no-confidence baseline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/distill_outputs_paper_kl_ce_no_confidence_old"),
    )
    return parser.parse_args()


def class_names_from_map(label_to_int):
    return [label for label, _ in sorted(label_to_int.items(), key=lambda item: item[1])]


def build_tcn_teacher_model(input_shape, num_classes):
    look_back, dim = input_shape
    inp = keras.Input((look_back, dim), name="vibration_input")
    x = keras.layers.Dense(32, activation="relu", name="teacher_embedding")(inp)
    x = TCN(
        seq_len=look_back,
        filters_list=[16, 32, 64],
        kernel_size_list=[3, 5, 7],
        name="teacher_tcn",
    )(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, name="teacher_feature")(x)
    out = keras.layers.Dense(num_classes, activation="softmax", name="teacher_probs")(x)
    return keras.Model(inp, out, name="tcn_vibration_teacher")


def load_tcn_teacher_model(path):
    return keras.models.load_model(
        path,
        compile=False,
        custom_objects={"TCN": TCN},
    )


def build_audio_student(input_shape, num_classes):
    return build_audio_student_model(
        input_shape=input_shape,
        num_classes=num_classes,
        input_name="spectrogram_input",
        model_name="audio_student_spectrogram",
        prefix="student",
    )


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


def resolve_sync_overrides(paths, args):
    sync_block = paths.get("sync", {}) if isinstance(paths.get("sync", {}), dict) else {}
    audio_offset_sec = sync_block.get("audio_offset_sec", paths.get("audio_offset_sec", args.audio_offset_sec))
    vibration_offset_sec = sync_block.get(
        "vibration_offset_sec",
        paths.get("vibration_offset_sec", args.vibration_offset_sec),
    )
    shared_duration_sec = sync_block.get(
        "shared_duration_sec",
        paths.get("shared_duration_sec", args.shared_duration_sec),
    )
    return (
        float(audio_offset_sec),
        float(vibration_offset_sec),
        None if shared_duration_sec is None else float(shared_duration_sec),
    )


def build_grouped_dataset_for_tcn_teacher(config, args, teacher_window_size):
    audio_specs_batches = []
    teacher_window_groups = []
    labels = []
    dataset_stats = []
    label_to_int = {label: index for index, label in enumerate(sorted(config))}

    audio_window = max(1, round(args.event_duration_sec * args.audio_target_sr))
    audio_hop = max(1, round(args.hop_sec * args.audio_target_sr))

    for label, paths in config.items():
        audio_path = Path(paths["audio"])
        vibration_path = Path(paths["vibration"])
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing audio file: {audio_path}")
        if not vibration_path.exists():
            raise FileNotFoundError(f"Missing vibration file: {vibration_path}")

        _, audio_signal = load_audio_signal(
            audio_path,
            target_sr=args.audio_target_sr,
            highpass_hz=args.audio_highpass_hz,
        )
        numeric_frame, time_seconds = load_vibration_frame(vibration_path)
        audio_offset_sec, vibration_offset_sec, shared_duration_override = resolve_sync_overrides(
            paths, args
        )

        sync = synchronized_time_range(
            audio_num_samples=len(audio_signal),
            audio_sample_rate=args.audio_target_sr,
            vibration_time_seconds=time_seconds,
            audio_offset_sec=audio_offset_sec,
            vibration_offset_sec=vibration_offset_sec,
            shared_duration_sec=shared_duration_override,
        )
        aligned_audio = audio_signal[sync["audio_start_sample"] : sync["audio_end_sample"]]
        vibration_values, aligned_vibration_times = crop_vibration_to_time_range(
            numeric_frame,
            time_seconds,
            start_sec=sync["vibration_start_sec"],
            end_sec=sync["vibration_end_sec"],
        )
        vibration_values = zscore_normalize(vibration_values)

        if len(aligned_audio) < audio_window or len(vibration_values) == 0:
            continue

        audio_windows = sliding_windows_1d(aligned_audio, audio_window, audio_hop)
        if len(audio_windows) == 0:
            continue

        start_times_sec = (np.arange(len(audio_windows), dtype=np.float32) * audio_hop) / float(args.audio_target_sr)
        end_times_sec = start_times_sec + (audio_window / float(args.audio_target_sr))

        valid_audio_specs = []
        valid_teacher_groups = []
        for audio_window_signal, start_time_sec, end_time_sec in zip(
            audio_windows,
            start_times_sec,
            end_times_sec,
        ):
            teacher_window = resample_vibration_interval(
                values=vibration_values,
                time_seconds=aligned_vibration_times,
                start_sec=float(start_time_sec),
                end_sec=float(end_time_sec),
                target_length=teacher_window_size,
            )
            if len(teacher_window) == 0:
                continue
            teacher_windows = teacher_window[np.newaxis, ...]
            audio_spec = compute_log_spectrograms(
                audio_window_signal[np.newaxis, :],
                sample_rate=args.audio_target_sr,
                nperseg=args.nperseg,
                noverlap=args.noverlap,
                max_freq_hz=args.max_freq_hz,
            )
            if len(audio_spec) == 0:
                continue
            valid_audio_specs.append(audio_spec[0])
            valid_teacher_groups.append(teacher_windows)

        if not valid_audio_specs:
            continue

        paired_count = len(valid_audio_specs)
        audio_specs_batches.append(np.stack(valid_audio_specs, axis=0))
        teacher_window_groups.extend(valid_teacher_groups)
        labels.append(np.full((paired_count,), label_to_int[label], dtype=np.int32))
        dataset_stats.append(
            {
                "label": label,
                "audio_path": str(audio_path),
                "vibration_path": str(vibration_path),
                "paired_window_count": int(paired_count),
                "shared_duration_sec": float(sync["shared_duration_sec"]),
                "audio_duration_sec": float(sync["audio_duration_sec"]),
                "vibration_duration_sec": float(sync["vibration_duration_sec"]),
                "audio_offset_sec": float(sync["audio_offset_sec"]),
                "vibration_offset_sec": float(sync["vibration_offset_sec"]),
                "teacher_window_size": int(teacher_window_size),
                "avg_teacher_windows_per_audio": float(
                    np.mean([len(group) for group in valid_teacher_groups])
                ),
            }
        )

    if not audio_specs_batches:
        raise ValueError("No paired samples were built for the TCN teacher distillation run.")

    x_audio = np.concatenate(audio_specs_batches, axis=0)
    y = np.concatenate(labels, axis=0)
    return x_audio, teacher_window_groups, y, label_to_int, dataset_stats


def flatten_teacher_groups(teacher_window_groups, labels):
    if not teacher_window_groups:
        return np.empty((0, 0, 0), dtype=np.float32), np.empty((0,), dtype=np.int32)
    flat_windows = np.concatenate(teacher_window_groups, axis=0)
    repeated_labels = np.concatenate(
        [
            np.full((len(group),), int(label), dtype=np.int32)
            for group, label in zip(teacher_window_groups, labels)
        ],
        axis=0,
    )
    return flat_windows.astype(np.float32), repeated_labels


def aggregate_teacher_probabilities(teacher, teacher_window_groups, batch_size):
    if not teacher_window_groups:
        return np.empty((0, int(teacher.output_shape[-1])), dtype=np.float32)
    flat_windows = np.concatenate(teacher_window_groups, axis=0)
    flat_probs = teacher.predict(flat_windows, batch_size=batch_size, verbose=0)
    counts = [len(group) for group in teacher_window_groups]
    split_points = np.cumsum(counts[:-1], dtype=np.int32)
    grouped_probs = np.split(flat_probs, split_points) if len(split_points) > 0 else [flat_probs]
    mean_probs = np.stack([probs.mean(axis=0) for probs in grouped_probs], axis=0)
    return mean_probs.astype(np.float32)


def build_tf_dataset(audio_data, teacher_soft_data, labels, batch_size, training):
    dataset = tf.data.Dataset.from_tensor_slices(((audio_data, teacher_soft_data), labels))
    if training:
        dataset = dataset.shuffle(len(labels), reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


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


class DistillationTrainerTCN(keras.Model):
    def __init__(
        self,
        teacher,
        student,
        alpha,
        temperature,
        ce_class_weights=None,
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
        self.ce_class_weights = None if ce_class_weights is None else tf.constant(
            np.asarray(ce_class_weights, dtype=np.float32)
        )
        self.focus_margin_label_index = (
            None if focus_margin_label_index is None else int(focus_margin_label_index)
        )
        self.focus_margin_rival_index = (
            None if focus_margin_rival_index is None else int(focus_margin_rival_index)
        )
        self.focus_margin_value = float(focus_margin_value)
        self.focus_margin_weight = float(focus_margin_weight)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.ce_tracker = keras.metrics.Mean(name="ce_loss")
        self.kl_tracker = keras.metrics.Mean(name="kl_loss")
        self.margin_tracker = keras.metrics.Mean(name="margin_loss")
        self.acc_metric = keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.teacher.trainable = False

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.ce_tracker,
            self.kl_tracker,
            self.margin_tracker,
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

        total_per_sample = (1.0 - self.alpha) * ce_per_sample + self.alpha * kl_per_sample
        total_loss = tf.reduce_mean(total_per_sample) + self.focus_margin_weight * margin_loss
        ce_loss = tf.reduce_mean(ce_per_sample)
        kl_loss = tf.reduce_mean(kl_per_sample)
        return total_loss, ce_loss, kl_loss, margin_loss, student_logits

    def train_step(self, data):
        (audio_batch, teacher_soft_batch), labels = data
        with tf.GradientTape() as tape:
            total_loss, ce_loss, kl_loss, margin_loss, student_logits = self._compute_losses(
                audio_batch, teacher_soft_batch, labels, training=True
            )
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        self.loss_tracker.update_state(total_loss)
        self.ce_tracker.update_state(ce_loss)
        self.kl_tracker.update_state(kl_loss)
        self.margin_tracker.update_state(margin_loss)
        self.acc_metric.update_state(labels, student_logits)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        (audio_batch, teacher_soft_batch), labels = data
        total_loss, ce_loss, kl_loss, margin_loss, student_logits = self._compute_losses(
            audio_batch, teacher_soft_batch, labels, training=False
        )
        self.loss_tracker.update_state(total_loss)
        self.ce_tracker.update_state(ce_loss)
        self.kl_tracker.update_state(kl_loss)
        self.margin_tracker.update_state(margin_loss)
        self.acc_metric.update_state(labels, student_logits)
        return {metric.name: metric.result() for metric in self.metrics}


def evaluate_teacher(teacher, teacher_window_groups_test, y_test, label_to_int, output_dir, batch_size):
    class_names = class_names_from_map(label_to_int)
    probs = aggregate_teacher_probabilities(teacher, teacher_window_groups_test, batch_size=batch_size)
    y_pred = np.argmax(probs, axis=1)
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    loss_value = float(loss_fn(y_test, probs).numpy())
    accuracy_value = float(np.mean(y_pred == y_test))
    metrics = {"loss": loss_value, "accuracy": accuracy_value}
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    save_confusion_matrix_figure(
        y_test,
        y_pred,
        class_names,
        output_dir / "teacher_confusion_matrix.png",
    )
    with (output_dir / "teacher_classification_report.txt").open("w", encoding="utf-8") as handle:
        handle.write(report)
    return metrics, report


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.random_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paired_config = select_labels(load_config(args.config), args.labels)
    teacher_model_path = Path(args.teacher_model)
    if not teacher_model_path.exists():
        raise FileNotFoundError(
            f"Teacher model not found at {teacher_model_path}. "
            "Point --teacher-model to a valid teacher_vibration.keras checkpoint."
        )

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
    distiller = DistillationTrainerTCN(
        teacher=teacher,
        student=student,
        alpha=args.alpha,
        temperature=args.temperature,
        ce_class_weights=ce_class_weights,
        focus_margin_label_index=focus_margin_label_index,
        focus_margin_rival_index=focus_margin_rival_index,
        focus_margin_value=args.focus_margin_value,
        focus_margin_weight=args.focus_margin_weight,
    )
    distiller.compile(optimizer=build_optimizer())
    train_ds = build_tf_dataset(
        x_audio_train, teacher_soft_train, y_train, args.batch_size, training=True
    )
    val_ds = build_tf_dataset(
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
        f"No-Confidence Student Distillation Curves (lambda={args.alpha:.2f}, T={args.temperature:.1f})",
        ["accuracy", "loss"],
    )
    plot_history(
        student_history,
        args.output_dir / "student_ce_kl.png",
        f"No-Confidence Student CE and KL (lambda={args.alpha:.2f}, T={args.temperature:.1f})",
        ["ce_loss", "kl_loss", "margin_loss"],
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
    with (args.output_dir / "student_classification_report.txt").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write(student_report)
    student.save(args.output_dir / "student_audio_only.keras")

    with (args.output_dir / "label_to_int.json").open("w", encoding="utf-8") as handle:
        json.dump(label_to_int, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "paired_dataset_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_stats, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "distill_run_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "distillation_variant": "no_confidence",
                "teacher_model": str(teacher_model_path),
                "teacher_window_size": int(teacher_window_size),
                "alpha": args.alpha,
                "temperature": args.temperature,
                "student_class_weight": (
                    None
                    if ce_class_weights is None
                    else {
                        label: float(ce_class_weights[index])
                        for label, index in sorted(label_to_int.items(), key=lambda item: item[1])
                    }
                ),
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

    print("No-confidence teacher vibration classification report:")
    print(teacher_report)
    print("No-confidence student audio classification report:")
    print(student_report)


if __name__ == "__main__":
    main()
