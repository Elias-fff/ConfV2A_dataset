import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras

from audio_student_model import (
    build_audio_student_model,
    build_optimizer,
    compute_log_spectrograms,
    load_audio_signal,
)
from event_windows import load_vibration_frame, zscore_normalize
from layers.tcn import TCN


DEFAULT_PAIRED_CONFIG = {
    "bottle_1000": {
        "audio": "data/raw/sensor_plot_project/bottle_1/1.wav",
        "vibration": "data/processed/bottle_1000.csv",
    },
    "bottle_500": {
        "audio": "data/raw/sensor_plot_project/bottle_500/500ml.wav",
        "vibration": "data/processed/bottle_500.csv",
    },
    "bottle_can_330": {
        "audio": "data/raw/sensor_plot_project/bottle_yi_330/yi_330.wav",
        "vibration": "data/processed/bottle_can_330.csv",
    },
    "bottle_can_500": {
        "audio": "data/raw/sensor_plot_project/bottle_yi_500/yi_500.wav",
        "vibration": "data/processed/bottle_can_500.csv",
    },
    "error": {
        "audio": "data/raw/sensor_plot_project/error/error.wav",
        "vibration": "data/processed/error.csv",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an audio spectrogram student with a vibration teacher."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON file that overrides the paired modality paths.",
    )
    parser.add_argument("--audio-target-sr", type=int, default=16000)
    parser.add_argument("--event-duration-sec", type=float, default=1.5)
    parser.add_argument("--hop-sec", type=float, default=0.3)
    parser.add_argument("--audio-highpass-hz", type=float, default=80.0)
    parser.add_argument("--nperseg", type=int, default=512)
    parser.add_argument("--noverlap", type=int, default=384)
    parser.add_argument("--max-freq-hz", type=float, default=6000.0)
    parser.add_argument("--teacher-epochs", type=int, default=60)
    parser.add_argument("--student-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--alpha-list",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of lambda(alpha) values to sweep.",
    )
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--student-init-model",
        type=Path,
        default=None,
        help="Optional pretrained audio student .keras model used to initialize the student before distillation or student-only finetuning.",
    )
    parser.add_argument(
        "--student-learning-rate",
        type=float,
        default=None,
        help="Optional student learning rate override. If omitted, finetuning uses 1e-4 and training from scratch uses 1e-3.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional subset of labels to include, e.g. bottle_1000 bottle_500 bottle_can_330 bottle_can_500.",
    )
    parser.add_argument(
        "--student-only",
        action="store_true",
        help="Train the audio student on the synchronized paired dataset without KL distillation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/distill_outputs"),
        help="Directory used to store the teacher, student, and label map.",
    )
    return parser.parse_args()


def load_config(path):
    if path is None:
        return DEFAULT_PAIRED_CONFIG
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_labels(config, labels):
    if labels is None:
        return dict(config)

    requested = list(dict.fromkeys(labels))
    missing = [label for label in requested if label not in config]
    if missing:
        available = ", ".join(sorted(config))
        raise ValueError(
            f"Unknown labels requested: {missing}. Available labels: {available}"
        )

    selected = {label: config[label] for label in requested}
    if len(selected) < 2:
        raise ValueError("At least two labels are required to run classification.")
    return selected


def resample_vibration_sequence(numeric_frame, source_time_seconds, target_length):
    values = numeric_frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    values = zscore_normalize(values)
    if len(values) == 0 or target_length <= 0:
        return np.empty((0, values.shape[-1] if values.ndim == 2 else 0), dtype=np.float32)
    if len(values) == target_length:
        return values.astype(np.float32)

    if len(source_time_seconds) == len(values) and len(source_time_seconds) > 1:
        duration = float(source_time_seconds[-1] - source_time_seconds[0])
        if duration > 0:
            source_grid = source_time_seconds - source_time_seconds[0]
            target_grid = np.linspace(0.0, duration, num=target_length, dtype=np.float32)
            resampled = [
                np.interp(target_grid, source_grid, values[:, dim]).astype(np.float32)
                for dim in range(values.shape[1])
            ]
            return np.stack(resampled, axis=-1)

    source_positions = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    resampled = [
        np.interp(target_positions, source_positions, values[:, dim]).astype(np.float32)
        for dim in range(values.shape[1])
    ]
    return np.stack(resampled, axis=-1)


def sliding_windows_aligned(audio_signal, vibration_signal, window_audio, hop_audio):
    if len(audio_signal) < window_audio or len(vibration_signal) < window_audio:
        return (
            np.empty((0, window_audio), dtype=np.float32),
            np.empty((0, window_audio, vibration_signal.shape[-1]), dtype=np.float32),
        )

    audio_windows = []
    vibration_windows = []
    max_start = min(len(audio_signal), len(vibration_signal)) - window_audio
    for start in range(0, max_start + 1, hop_audio):
        audio_windows.append(audio_signal[start : start + window_audio])
        vibration_windows.append(vibration_signal[start : start + window_audio])

    return (
        np.asarray(audio_windows, dtype=np.float32),
        np.asarray(vibration_windows, dtype=np.float32),
    )


def build_paired_dataset(config, args):
    audio_batches = []
    vibration_batches = []
    labels = []
    dataset_stats = []
    label_to_int = {label: index for index, label in enumerate(sorted(config))}

    for label, paths in config.items():
        audio_path = Path(paths["audio"])
        vibration_path = Path(paths["vibration"])
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing audio file: {audio_path}")
        if not vibration_path.exists():
            raise FileNotFoundError(f"Missing vibration file: {vibration_path}")

        audio_sr, audio_signal = load_audio_signal(
            audio_path,
            target_sr=args.audio_target_sr,
            highpass_hz=args.audio_highpass_hz,
        )
        numeric_frame, vibration_times = load_vibration_frame(vibration_path)
        vibration_signal = resample_vibration_sequence(
            numeric_frame,
            vibration_times,
            target_length=len(audio_signal),
        )

        window_audio = max(1, round(args.event_duration_sec * args.audio_target_sr))
        hop_audio = max(1, round(args.hop_sec * args.audio_target_sr))
        paired_audio, paired_vibration = sliding_windows_aligned(
            audio_signal=audio_signal,
            vibration_signal=vibration_signal,
            window_audio=window_audio,
            hop_audio=hop_audio,
        )
        if len(paired_audio) == 0:
            continue

        audio_specs = compute_log_spectrograms(
            paired_audio,
            sample_rate=args.audio_target_sr,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
            max_freq_hz=args.max_freq_hz,
        )
        if len(audio_specs) == 0:
            continue

        paired_count = min(len(audio_specs), len(paired_vibration))
        audio_batches.append(audio_specs[:paired_count])
        vibration_batches.append(paired_vibration[:paired_count])
        labels.append(
            np.full((paired_count,), label_to_int[label], dtype=np.int32)
        )
        dataset_stats.append(
            {
                "label": label,
                "audio_path": str(audio_path),
                "vibration_path": str(vibration_path),
                "paired_window_count": int(paired_count),
            }
        )

    if not audio_batches:
        raise ValueError("No paired samples were built. Check the config paths and synchronization settings.")

    x_audio = np.concatenate(audio_batches, axis=0)
    x_vibration = np.concatenate(vibration_batches, axis=0)
    y = np.concatenate(labels, axis=0)
    return x_audio, x_vibration, y, label_to_int, dataset_stats


def build_vibration_teacher(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape, name="vibration_input")
    x = keras.layers.Dense(32, activation="relu", name="teacher_embedding")(inputs)
    x = TCN(
        seq_len=input_shape[0],
        filters_list=[16, 32, 64],
        kernel_size_list=[3, 5, 7],
        name="teacher_tcn",
    )(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    features = keras.layers.Dense(32, name="teacher_feature")(x)
    logits = keras.layers.Dense(num_classes, name="teacher_logits")(features)
    return keras.Model(inputs=inputs, outputs=logits, name="vibration_teacher")


def build_audio_student(input_shape, num_classes):
    return build_audio_student_model(
        input_shape=input_shape,
        num_classes=num_classes,
        input_name="audio_input",
        model_name="audio_student_spectrogram",
        prefix="student",
    )


def resolve_student_learning_rate(args):
    if args.student_learning_rate is not None:
        return float(args.student_learning_rate)
    return 1e-4 if args.student_init_model is not None else 1e-3


def load_or_build_audio_student(args, input_shape, num_classes):
    if args.student_init_model is None:
        return build_audio_student(input_shape, num_classes)

    init_path = Path(args.student_init_model)
    if not init_path.exists():
        raise FileNotFoundError(f"Missing student init model: {init_path}")

    student = keras.models.load_model(init_path, compile=False)
    expected_input_shape = tuple(input_shape)
    actual_input_shape = tuple(student.input_shape[1:])
    if actual_input_shape != expected_input_shape:
        raise ValueError(
            f"Student init model input shape {actual_input_shape} does not match expected {expected_input_shape}."
        )

    output_units = student.output_shape[-1]
    if int(output_units) != int(num_classes):
        raise ValueError(
            f"Student init model outputs {output_units} classes, but this run expects {num_classes}."
        )

    return student


def plot_history(history, output_path, title, keys):
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 5))
    for key in keys:
        if key in history.history:
            plt.plot(history.history[key], label=key)
        val_key = f"val_{key}"
        if val_key in history.history:
            plt.plot(history.history[val_key], label=val_key)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_confusion_matrix_figure(y_true, y_pred, class_names, output_path):
    matrix = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_labels = [str(index + 1) for index in range(len(class_names))]
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        xlabel="Predicted label",
        ylabel="Truth label",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    threshold = 0.5
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > threshold else "#222222"
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
            )

    ax.set_title("Normalized Confusion Matrix", pad=12)
    fig.text(
        0.5,
        0.01,
        "Class index mapping: " + ", ".join(
            f"{idx + 1}={name}" for idx, name in enumerate(class_names)
        ),
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


class DistillationTrainer(keras.Model):
    def __init__(self, teacher, student, alpha, temperature):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.temperature = temperature
        self.student_loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
        self.distill_loss_fn = keras.losses.KLDivergence()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.ce_tracker = keras.metrics.Mean(name="ce_loss")
        self.kl_tracker = keras.metrics.Mean(name="kl_loss")
        self.acc_metric = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

        self.teacher_probe = keras.Model(
            inputs=teacher.input,
            outputs=teacher.output,
        )
        self.student_probe = keras.Model(
            inputs=student.input,
            outputs=student.output,
        )
        self.teacher.trainable = False

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.ce_tracker,
            self.kl_tracker,
            self.acc_metric,
        ]

    def _compute_losses(self, audio_batch, vibration_batch, labels, training):
        teacher_logits = self.teacher_probe(vibration_batch, training=False)
        student_logits = self.student_probe(audio_batch, training=training)

        ce_loss = self.student_loss_fn(labels, student_logits)
        teacher_soft = tf.nn.softmax(teacher_logits / self.temperature, axis=-1)
        student_soft = tf.nn.softmax(student_logits / self.temperature, axis=-1)
        kl_loss = (
            self.distill_loss_fn(teacher_soft, student_soft)
            * self.temperature
            * self.temperature
        )
        total_loss = (1.0 - self.alpha) * ce_loss + self.alpha * kl_loss
        return total_loss, ce_loss, kl_loss, student_logits

    def train_step(self, data):
        (audio_batch, vibration_batch), labels = data
        with tf.GradientTape() as tape:
            total_loss, ce_loss, kl_loss, student_logits = (
                self._compute_losses(
                    audio_batch, vibration_batch, labels, training=True
                )
            )
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        self.loss_tracker.update_state(total_loss)
        self.ce_tracker.update_state(ce_loss)
        self.kl_tracker.update_state(kl_loss)
        self.acc_metric.update_state(labels, student_logits)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        (audio_batch, vibration_batch), labels = data
        total_loss, ce_loss, kl_loss, student_logits = (
            self._compute_losses(audio_batch, vibration_batch, labels, training=False)
        )

        self.loss_tracker.update_state(total_loss)
        self.ce_tracker.update_state(ce_loss)
        self.kl_tracker.update_state(kl_loss)
        self.acc_metric.update_state(labels, student_logits)
        return {metric.name: metric.result() for metric in self.metrics}


def build_tf_dataset(audio_data, vibration_data, labels, batch_size, training):
    dataset = tf.data.Dataset.from_tensor_slices(((audio_data, vibration_data), labels))
    if training:
        dataset = dataset.shuffle(len(labels), reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_audio_only_dataset(audio_data, labels, batch_size, training):
    dataset = tf.data.Dataset.from_tensor_slices((audio_data, labels))
    if training:
        dataset = dataset.shuffle(len(labels), reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def alpha_dir_name(alpha):
    return f"alpha_{alpha:.2f}".replace(".", "_")


def train_one_student_only_run(
    args,
    x_audio_train,
    x_audio_val,
    x_audio_test,
    y_train,
    y_val,
    y_test,
    label_to_int,
    output_dir,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(label_to_int)
    student = load_or_build_audio_student(args, x_audio_train.shape[1:], num_classes)
    student_lr = resolve_student_learning_rate(args)
    student.compile(
        optimizer=build_optimizer(learning_rate=student_lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    train_ds = build_audio_only_dataset(
        x_audio_train, y_train, args.batch_size, training=True
    )
    val_ds = build_audio_only_dataset(
        x_audio_val, y_val, args.batch_size, training=False
    )
    student_history = student.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.student_epochs,
        verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=12,
                restore_best_weights=True,
            )
        ],
    )

    test_metrics = student.evaluate(
        x_audio_test, y_test, batch_size=args.batch_size, verbose=0, return_dict=True
    )
    print("Student-only synchronized-window audio test metrics:", test_metrics)

    class_names = [label for label, _ in sorted(label_to_int.items(), key=lambda item: item[1])]
    test_logits = student.predict(x_audio_test, batch_size=args.batch_size, verbose=0)
    y_pred = np.argmax(test_logits, axis=1)
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print("Student-only synchronized-window audio classification report:")
    print(report)

    plot_history(
        student_history,
        output_dir / "student_only_accuracy_loss.png",
        "Student-Only Training Curves",
        ["accuracy", "loss"],
    )
    save_confusion_matrix_figure(
        y_test,
        y_pred,
        class_names,
        output_dir / "student_only_confusion_matrix.png",
    )

    student.save(output_dir / "student_audio_only.keras")
    with (output_dir / "student_only_infer_meta.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "student_only": True,
                "student_init_model": (
                    str(args.student_init_model) if args.student_init_model is not None else None
                ),
                "student_learning_rate": student_lr,
                "audio_target_sr": args.audio_target_sr,
                "event_duration_sec": args.event_duration_sec,
                "hop_sec": args.hop_sec,
                "audio_highpass_hz": args.audio_highpass_hz,
                "nperseg": args.nperseg,
                "noverlap": args.noverlap,
                "max_freq_hz": args.max_freq_hz,
                "student_infer_hop_sec": 0.3,
                "label_to_int": label_to_int,
                "student_model": "student_audio_only.keras",
                "student_input_shape": list(x_audio_train.shape[1:]),
                "student_input_type": "log_spectrogram",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    with (output_dir / "student_only_classification_report.txt").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write(report)

    return {
        "mode": "student_only",
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
    }


def train_one_student_run(
    teacher,
    alpha,
    args,
    x_audio_train,
    x_audio_val,
    x_audio_test,
    x_vibration_train,
    x_vibration_val,
    y_train,
    y_val,
    y_test,
    label_to_int,
    output_dir,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(label_to_int)
    student = load_or_build_audio_student(args, x_audio_train.shape[1:], num_classes)
    student_lr = resolve_student_learning_rate(args)
    distiller = DistillationTrainer(
        teacher=teacher,
        student=student,
        alpha=alpha,
        temperature=args.temperature,
    )
    distiller.compile(optimizer=build_optimizer(learning_rate=student_lr))

    train_ds = build_tf_dataset(
        x_audio_train, x_vibration_train, y_train, args.batch_size, training=True
    )
    val_ds = build_tf_dataset(
        x_audio_val, x_vibration_val, y_val, args.batch_size, training=False
    )
    student_history = distiller.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.student_epochs,
        verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=12,
                restore_best_weights=True,
            )
        ],
    )

    student.compile(
        optimizer=build_optimizer(learning_rate=student_lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    test_metrics = student.evaluate(
        x_audio_test, y_test, batch_size=args.batch_size, verbose=0, return_dict=True
    )
    print(f"Student-only audio test metrics for alpha={alpha:.2f}:", test_metrics)

    class_names = [label for label, _ in sorted(label_to_int.items(), key=lambda item: item[1])]
    test_logits = student.predict(x_audio_test, batch_size=args.batch_size, verbose=0)
    y_pred = np.argmax(test_logits, axis=1)
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(f"Student-only audio classification report for alpha={alpha:.2f}:")
    print(report)

    plot_history(
        student_history,
        output_dir / "student_accuracy_loss.png",
        f"Student Distillation Curves (alpha={alpha:.2f})",
        ["accuracy", "loss"],
    )
    plot_history(
        student_history,
        output_dir / "student_ce_kl.png",
        f"Student CE and KL Loss (alpha={alpha:.2f})",
        ["ce_loss", "kl_loss"],
    )
    save_confusion_matrix_figure(
        y_test,
        y_pred,
        class_names,
        output_dir / "student_confusion_matrix.png",
    )

    student.save(output_dir / "student_audio_only.keras")
    with (output_dir / "student_infer_meta.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "alpha": alpha,
                "temperature": args.temperature,
                "student_init_model": (
                    str(args.student_init_model) if args.student_init_model is not None else None
                ),
                "student_learning_rate": student_lr,
                "audio_target_sr": args.audio_target_sr,
                "event_duration_sec": args.event_duration_sec,
                "hop_sec": args.hop_sec,
                "audio_highpass_hz": args.audio_highpass_hz,
                "nperseg": args.nperseg,
                "noverlap": args.noverlap,
                "max_freq_hz": args.max_freq_hz,
                "student_infer_hop_sec": 0.3,
                "label_to_int": label_to_int,
                "student_model": "student_audio_only.keras",
                "student_input_shape": list(x_audio_train.shape[1:]),
                "student_input_type": "log_spectrogram",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    with (output_dir / "student_classification_report.txt").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write(report)

    return {
        "alpha": alpha,
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
    }


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.random_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paired_config = select_labels(load_config(args.config), args.labels)
    x_audio, x_vibration, y, label_to_int, dataset_stats = build_paired_dataset(
        paired_config, args
    )

    split_payload = train_test_split(
        x_audio,
        x_vibration,
        y,
        test_size=0.3,
        random_state=args.random_state,
        stratify=y,
    )
    (
        x_audio_train,
        x_audio_temp,
        x_vibration_train,
        x_vibration_temp,
        y_train,
        y_temp,
    ) = split_payload

    split_payload = train_test_split(
        x_audio_temp,
        x_vibration_temp,
        y_temp,
        test_size=0.5,
        random_state=args.random_state,
        stratify=y_temp,
    )
    (
        x_audio_val,
        x_audio_test,
        x_vibration_val,
        x_vibration_test,
        y_val,
        y_test,
    ) = split_payload

    num_classes = len(label_to_int)
    teacher = build_vibration_teacher(x_vibration_train.shape[1:], num_classes)
    teacher.compile(
        optimizer=build_optimizer(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    teacher_history = teacher.fit(
        x_vibration_train,
        y_train,
        validation_data=(x_vibration_val, y_val),
        epochs=args.teacher_epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            )
        ],
    )

    plot_history(
        teacher_history,
        args.output_dir / "teacher_accuracy_loss.png",
        "Teacher Training Curves",
        ["accuracy", "loss"],
    )

    teacher_test_metrics = teacher.evaluate(
        x_vibration_test,
        y_test,
        batch_size=args.batch_size,
        verbose=0,
        return_dict=True,
    )
    print("Teacher vibration test metrics:", teacher_test_metrics)

    class_names = [
        label for label, _ in sorted(label_to_int.items(), key=lambda item: item[1])
    ]
    teacher_test_logits = teacher.predict(
        x_vibration_test, batch_size=args.batch_size, verbose=0
    )
    teacher_y_pred = np.argmax(teacher_test_logits, axis=1)
    teacher_report = classification_report(
        y_test,
        teacher_y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print("Teacher vibration classification report:")
    print(teacher_report)
    save_confusion_matrix_figure(
        y_test,
        teacher_y_pred,
        class_names,
        args.output_dir / "teacher_confusion_matrix.png",
    )

    teacher.save(args.output_dir / "teacher_vibration.keras")
    with (args.output_dir / "label_to_int.json").open("w", encoding="utf-8") as handle:
        json.dump(label_to_int, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "paired_config.json").open("w", encoding="utf-8") as handle:
        json.dump(paired_config, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "paired_dataset_stats.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(dataset_stats, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "teacher_classification_report.txt").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write(teacher_report)
    alpha_values = args.alpha_list if args.alpha_list is not None else [args.alpha]
    with (args.output_dir / "distill_run_meta.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "student_only": args.student_only,
                "alpha_values": alpha_values,
                "teacher_test_loss": float(teacher_test_metrics["loss"]),
                "teacher_test_accuracy": float(teacher_test_metrics["accuracy"]),
                "temperature": args.temperature,
                "student_init_model": (
                    str(args.student_init_model) if args.student_init_model is not None else None
                ),
                "student_learning_rate": resolve_student_learning_rate(args),
                "audio_target_sr": args.audio_target_sr,
                "event_duration_sec": args.event_duration_sec,
                "hop_sec": args.hop_sec,
                "audio_highpass_hz": args.audio_highpass_hz,
                "nperseg": args.nperseg,
                "noverlap": args.noverlap,
                "max_freq_hz": args.max_freq_hz,
                "student_infer_hop_sec": 0.3,
                "label_to_int": label_to_int,
                "student_input_type": "log_spectrogram",
                "student_input_shape": list(x_audio_train.shape[1:]),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    if args.student_only:
        student_only_summary = train_one_student_only_run(
            args=args,
            x_audio_train=x_audio_train,
            x_audio_val=x_audio_val,
            x_audio_test=x_audio_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            label_to_int=label_to_int,
            output_dir=args.output_dir,
        )
        with (args.output_dir / "student_only_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(student_only_summary, handle, ensure_ascii=False, indent=2)
        print("Student-only summary:")
        print(pd.DataFrame([student_only_summary]).to_string(index=False))
        return

    summary_rows = []
    best_run = None

    for alpha in alpha_values:
        keras.utils.set_random_seed(args.random_state)
        run_dir = args.output_dir if len(alpha_values) == 1 else args.output_dir / alpha_dir_name(alpha)
        run_summary = train_one_student_run(
            teacher=teacher,
            alpha=alpha,
            args=args,
            x_audio_train=x_audio_train,
            x_audio_val=x_audio_val,
            x_audio_test=x_audio_test,
            x_vibration_train=x_vibration_train,
            x_vibration_val=x_vibration_val,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            label_to_int=label_to_int,
            output_dir=run_dir,
        )
        summary_rows.append(run_summary)
        if best_run is None or run_summary["test_accuracy"] > best_run["test_accuracy"]:
            best_run = run_summary

    summary_frame = pd.DataFrame(summary_rows).sort_values(
        by="test_accuracy", ascending=False
    )
    summary_frame.to_csv(args.output_dir / "alpha_sweep_summary.csv", index=False)
    with (args.output_dir / "best_alpha.json").open("w", encoding="utf-8") as handle:
        json.dump(best_run, handle, ensure_ascii=False, indent=2)
    print("Alpha sweep summary:")
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
