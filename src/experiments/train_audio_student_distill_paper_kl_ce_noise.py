import argparse
import csv
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
    resolve_sync_overrides,
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
    def __init__(self, teacher, student, alpha, temperature, confidence_mode, confidence_threshold):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.confidence_mode = str(confidence_mode)
        self.confidence_threshold = float(confidence_threshold)
        self.student_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.ce_tracker = keras.metrics.Mean(name="ce_loss")
        self.kl_tracker = keras.metrics.Mean(name="kl_loss")
        self.confidence_tracker = keras.metrics.Mean(name="confidence_weight")
        self.acc_metric = keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.teacher.trainable = False

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.ce_tracker,
            self.kl_tracker,
            self.confidence_tracker,
            self.acc_metric,
        ]

    def _compute_losses(self, audio_batch, teacher_soft_batch, labels, training):
        student_logits = self.student(audio_batch, training=training)
        ce_per_sample = keras.losses.sparse_categorical_crossentropy(
            labels, student_logits, from_logits=True
        )
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

        total_per_sample = (1.0 - self.alpha) * ce_per_sample + self.alpha * (
            kl_weight * kl_per_sample
        )
        total_loss = tf.reduce_mean(total_per_sample)
        ce_loss = tf.reduce_mean(ce_per_sample)
        kl_loss = tf.math.divide_no_nan(
            tf.reduce_sum(kl_weight * kl_per_sample),
            tf.reduce_sum(kl_weight),
        )
        confidence_weight = tf.reduce_mean(kl_weight)
        return total_loss, ce_loss, kl_loss, confidence_weight, student_logits

    def train_step(self, data):
        (audio_batch, teacher_soft_batch), labels = data
        with tf.GradientTape() as tape:
            total_loss, ce_loss, kl_loss, confidence_weight, student_logits = self._compute_losses(
                audio_batch, teacher_soft_batch, labels, training=True
            )
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        self.loss_tracker.update_state(total_loss)
        self.ce_tracker.update_state(ce_loss)
        self.kl_tracker.update_state(kl_loss)
        self.confidence_tracker.update_state(confidence_weight)
        self.acc_metric.update_state(labels, student_logits)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        (audio_batch, teacher_soft_batch), labels = data
        total_loss, ce_loss, kl_loss, confidence_weight, student_logits = self._compute_losses(
            audio_batch, teacher_soft_batch, labels, training=False
        )
        self.loss_tracker.update_state(total_loss)
        self.ce_tracker.update_state(ce_loss)
        self.kl_tracker.update_state(kl_loss)
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
    parser.add_argument("--teacher-window-size", type=int, default=300)
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
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--student-init-model",
        type=Path,
        default=DEFAULT_TIME_ALIGNED_AUDIO_STUDENT,
        help=(
            "Initial audio student checkpoint. Defaults to the time-aligned audio "
            "spectrogram noise-trained model. If the file is missing, the student is trained from scratch."
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
        default=Path("results/distill_outputs_paper_kl_ce_noise_old"),
    )
    parser.add_argument(
        "--noise-snr-db-list",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional SNR values in dB for post-training noisy test evaluation, "
            "e.g. 30 20 10 5 0. Noise is added to raw waveform test windows "
            "before recomputing log-spectrograms."
        ),
    )
    return parser.parse_args()


def snr_dir_name(snr_db):
    return f"snr_{snr_db:.1f}db".replace("-", "minus_").replace(".", "_")


def add_waveform_noise_at_snr(windows, snr_db, rng):
    windows = np.asarray(windows, dtype=np.float32)
    signal_power = np.mean(np.square(windows), axis=1, keepdims=True)
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise = rng.normal(loc=0.0, scale=1.0, size=windows.shape).astype(np.float32)
    noise_std = np.sqrt(np.maximum(noise_power, 1e-12)).astype(np.float32)
    return windows + noise * noise_std


def evaluate_student_on_inputs(student, x_eval, y_eval, class_names, output_dir, batch_size):
    metrics = student.evaluate(
        x_eval, y_eval, batch_size=batch_size, verbose=0, return_dict=True
    )
    logits = student.predict(x_eval, batch_size=batch_size, verbose=0)
    predictions = np.argmax(logits, axis=1)
    report = classification_report(
        y_eval,
        predictions,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    save_confusion_matrix_figure(
        y_eval,
        predictions,
        class_names,
        output_dir / "student_confusion_matrix.png",
    )
    with (output_dir / "student_classification_report.txt").open("w", encoding="utf-8") as handle:
        handle.write(report)
    return metrics, report


def build_valid_waveform_windows_for_tcn_teacher(config, args, teacher_window_size):
    waveform_batches = []
    labels = []
    label_to_int = {label: index for index, label in enumerate(sorted(config))}

    audio_window = max(1, round(args.event_duration_sec * args.audio_target_sr))
    audio_hop = max(1, round(args.hop_sec * args.audio_target_sr))

    for label, paths in config.items():
        audio_path = Path(paths["audio"])
        vibration_path = Path(paths["vibration"])
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
        start_times_sec = (np.arange(len(audio_windows), dtype=np.float32) * audio_hop) / float(
            args.audio_target_sr
        )
        end_times_sec = start_times_sec + (audio_window / float(args.audio_target_sr))

        valid_windows = []
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
            audio_spec = compute_log_spectrograms(
                audio_window_signal[np.newaxis, :],
                sample_rate=args.audio_target_sr,
                nperseg=args.nperseg,
                noverlap=args.noverlap,
                max_freq_hz=args.max_freq_hz,
            )
            if len(audio_spec) == 0:
                continue
            valid_windows.append(audio_window_signal.astype(np.float32))

        if not valid_windows:
            continue

        waveform_batches.append(np.stack(valid_windows, axis=0))
        labels.append(np.full((len(valid_windows),), label_to_int[label], dtype=np.int32))

    if not waveform_batches:
        raise ValueError("No waveform windows were built for noisy evaluation.")

    return np.concatenate(waveform_batches, axis=0), np.concatenate(labels, axis=0)


def split_waveform_test_windows(waveform_windows, labels, dataset_stats, args):
    if args.split_mode == "random":
        indices = np.arange(len(labels), dtype=np.int32)
        _, temp_indices, _, y_temp = train_test_split(
            indices,
            labels,
            test_size=0.3,
            random_state=args.random_state,
            stratify=labels,
        )
        _, test_indices, _, _ = train_test_split(
            temp_indices,
            y_temp,
            test_size=0.5,
            random_state=args.random_state,
            stratify=y_temp,
        )
        return waveform_windows[test_indices], labels[test_indices]

    x_test_batches = []
    y_test_batches = []
    start = 0
    for item in dataset_stats:
        count = int(item["paired_window_count"])
        end = start + count
        x_label = waveform_windows[start:end]
        y_label = labels[start:end]
        start = end

        train_count, val_count, test_count = split_count_triplet(count)
        val_end = train_count + val_count
        x_test_batches.append(x_label[val_end:val_end + test_count])
        y_test_batches.append(y_label[val_end:val_end + test_count])

    return np.concatenate(x_test_batches, axis=0), np.concatenate(y_test_batches, axis=0)


def evaluate_student_waveform_noise_sweep(
    student,
    paired_config,
    dataset_stats,
    teacher_window_size,
    y_test,
    class_names,
    args,
):
    if not args.noise_snr_db_list:
        return []

    rows = []
    rng = np.random.default_rng(args.random_state)
    noise_root = args.output_dir / "waveform_noise_snr_eval"
    noise_root.mkdir(parents=True, exist_ok=True)
    waveform_windows, waveform_labels = build_valid_waveform_windows_for_tcn_teacher(
        paired_config,
        args,
        teacher_window_size=teacher_window_size,
    )
    waveform_test, waveform_y_test = split_waveform_test_windows(
        waveform_windows,
        waveform_labels,
        dataset_stats,
        args,
    )
    if len(waveform_test) != len(y_test) or not np.array_equal(waveform_y_test, y_test):
        raise ValueError(
            "Waveform noisy-test split does not match the clean spectrogram test split."
        )

    for snr_db in args.noise_snr_db_list:
        noisy_waveforms = add_waveform_noise_at_snr(waveform_test, snr_db, rng)
        x_noisy = compute_log_spectrograms(
            noisy_waveforms,
            sample_rate=args.audio_target_sr,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
            max_freq_hz=args.max_freq_hz,
        )
        run_dir = noise_root / snr_dir_name(snr_db)
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics, report = evaluate_student_on_inputs(
            student,
            x_noisy,
            y_test,
            class_names,
            run_dir,
            args.batch_size,
        )
        row = {
            "snr_db": float(snr_db),
            "student_test_loss": float(metrics["loss"]),
            "student_test_accuracy": float(metrics["accuracy"]),
            "support": int(len(y_test)),
            "noise_domain": "waveform",
            "output_dir": str(run_dir),
        }
        rows.append(row)
        with (run_dir / "noise_eval_meta.json").open("w", encoding="utf-8") as handle:
            json.dump(row, handle, ensure_ascii=False, indent=2)
        print(f"Waveform-noisy test SNR={snr_db:g} dB:")
        print(report)

    with (noise_root / "noise_eval_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "snr_db",
                "student_test_loss",
                "student_test_accuracy",
                "support",
                "noise_domain",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    with (noise_root / "noise_eval_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    return rows


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
    class_names = class_names_from_map(label_to_int)
    student_metrics, student_report = evaluate_student_on_inputs(
        student,
        x_audio_test,
        y_test,
        class_names,
        args.output_dir,
        args.batch_size,
    )
    noise_eval_rows = evaluate_student_waveform_noise_sweep(
        student,
        paired_config,
        dataset_stats,
        teacher_window_size,
        y_test,
        class_names,
        args,
    )
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
                "teacher_model": str(teacher_model_path) if teacher_model_path is not None else "trained_in_script",
                "teacher_window_size": int(teacher_window_size),
                "alpha": args.alpha,
                "temperature": args.temperature,
                "confidence_mode": args.confidence_mode,
                "teacher_confidence_threshold": args.teacher_confidence_threshold,
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
                "noise_snr_db_list": (
                    None
                    if args.noise_snr_db_list is None
                    else [float(value) for value in args.noise_snr_db_list]
                ),
                "noise_domain": "waveform",
                "noise_eval": noise_eval_rows,
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
