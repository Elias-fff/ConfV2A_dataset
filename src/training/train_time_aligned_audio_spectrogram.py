import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import matplotlib
matplotlib.use("Agg")
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from audio_student_model import (
    build_audio_student_model,
    build_optimizer,
    compute_log_spectrograms,
    filter_windows_by_energy,
    load_audio_signal,
    sliding_windows_1d,
)
from event_windows import load_vibration_frame, synchronized_time_range
from train_audio_student_distill import (
    DEFAULT_PAIRED_CONFIG,
    load_config,
    plot_history,
    save_confusion_matrix_figure,
    select_labels,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a pure-audio spectrogram CNN with time windows aligned to vibration timestamps."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON file with paired {'audio': ..., 'vibration': ...} paths.",
    )
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--window-sec", type=float, default=0.8)
    parser.add_argument(
        "--hop-sec",
        type=float,
        default=0.35,
        help="Window hop in seconds. Default keeps light overlap while preserving time alignment.",
    )
    parser.add_argument("--highpass-hz", type=float, default=80.0)
    parser.add_argument("--nperseg", type=int, default=512)
    parser.add_argument("--noverlap", type=int, default=384)
    parser.add_argument("--max-freq-hz", type=float, default=6000.0)
    parser.add_argument(
        "--energy-quantile",
        type=float,
        default=0.35,
        help="RMS quantile filter applied after time alignment.",
    )
    parser.add_argument(
        "--max-windows-per-class",
        type=int,
        default=None,
        help="Optional cap after aligned-window filtering.",
    )
    parser.add_argument("--audio-offset-sec", type=float, default=0.0)
    parser.add_argument("--vibration-offset-sec", type=float, default=0.0)
    parser.add_argument("--shared-duration-sec", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        choices=["blocked_time", "random"],
        default="blocked_time",
        help="Use blocked_time to keep train/val/test ordered along each recording timeline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/time_aligned_audio_spectrogram_outputs"),
    )
    return parser.parse_args()


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


def class_names_from_map(label_to_int):
    return [label for label, _ in sorted(label_to_int.items(), key=lambda item: item[1])]


def build_audio_model(input_shape, num_classes):
    return build_audio_student_model(
        input_shape=input_shape,
        num_classes=num_classes,
        input_name="spectrogram_input",
        model_name="time_aligned_audio_spectrogram_cnn",
        prefix=None,
    )


def maybe_filter_aligned_windows(windows, args, rng):
    if args.energy_quantile is None:
        rms = np.sqrt(np.mean(np.square(windows), axis=1)) if len(windows) else np.empty((0,), dtype=np.float32)
        return windows, rms

    return filter_windows_by_energy(
        windows,
        energy_quantile=args.energy_quantile,
        max_windows=args.max_windows_per_class,
        rng=rng,
    )


def build_time_aligned_audio_dataset(config, args):
    rng = np.random.default_rng(args.random_state)
    class_windows = []
    labels = []
    label_to_int = {label: index for index, label in enumerate(sorted(config))}
    stats = []

    window_size = max(1, round(args.window_sec * args.target_sr))
    hop_size = max(1, round(args.hop_sec * args.target_sr))

    for label, paths in config.items():
        audio_path = Path(paths["audio"])
        vibration_path = Path(paths["vibration"])
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing audio file: {audio_path}")
        if not vibration_path.exists():
            raise FileNotFoundError(f"Missing vibration file: {vibration_path}")

        _, audio_signal = load_audio_signal(
            audio_path,
            target_sr=args.target_sr,
            highpass_hz=args.highpass_hz,
        )
        _numeric_frame, vibration_times = load_vibration_frame(vibration_path)
        audio_offset_sec, vibration_offset_sec, shared_duration_override = resolve_sync_overrides(
            paths, args
        )
        sync = synchronized_time_range(
            audio_num_samples=len(audio_signal),
            audio_sample_rate=args.target_sr,
            vibration_time_seconds=vibration_times,
            audio_offset_sec=audio_offset_sec,
            vibration_offset_sec=vibration_offset_sec,
            shared_duration_sec=shared_duration_override,
        )

        aligned_audio = audio_signal[sync["audio_start_sample"] : sync["audio_end_sample"]]
        raw_windows = sliding_windows_1d(aligned_audio, window_size, hop_size)
        kept_windows, kept_rms = maybe_filter_aligned_windows(raw_windows, args, rng)
        if len(kept_windows) == 0:
            continue

        specs = compute_log_spectrograms(
            kept_windows,
            sample_rate=args.target_sr,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
            max_freq_hz=args.max_freq_hz,
        )
        if len(specs) == 0:
            continue

        class_windows.append(specs)
        labels.append(np.full((len(specs),), label_to_int[label], dtype=np.int32))
        stats.append(
            {
                "label": label,
                "audio_path": str(audio_path),
                "vibration_path": str(vibration_path),
                "audio_duration_sec": float(sync["audio_duration_sec"]),
                "vibration_duration_sec": float(sync["vibration_duration_sec"]),
                "shared_duration_sec": float(sync["shared_duration_sec"]),
                "audio_offset_sec": float(sync["audio_offset_sec"]),
                "vibration_offset_sec": float(sync["vibration_offset_sec"]),
                "raw_window_count": int(len(raw_windows)),
                "kept_window_count": int(len(specs)),
                "mean_rms": float(np.mean(kept_rms)) if len(kept_rms) else 0.0,
                "window_sec": float(args.window_sec),
                "hop_sec": float(args.hop_sec),
            }
        )

    if not class_windows:
        raise ValueError("No aligned audio samples were built. Check the paired config and sync settings.")

    x_audio = np.concatenate(class_windows, axis=0)
    y = np.concatenate(labels, axis=0)
    return x_audio, y, label_to_int, stats


def validate_class_counts(y, label_to_int):
    counts = np.bincount(y, minlength=len(label_to_int))
    min_count = int(np.min(counts))
    if min_count < 7:
        int_to_label = {value: key for key, value in label_to_int.items()}
        readable = ", ".join(
            f"{int_to_label[index]}={int(count)}" for index, count in enumerate(counts)
        )
        raise ValueError(
            "Each class needs at least 7 windows for the default train/val/test split. "
            f"Current counts: {readable}."
        )


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


def split_dataset(x_audio, y, stats, args):
    if args.split_mode == "random":
        split_payload = train_test_split(
            x_audio,
            y,
            test_size=0.3,
            random_state=args.random_state,
            stratify=y,
        )
        x_train, x_temp, y_train, y_temp = split_payload

        split_payload = train_test_split(
            x_temp,
            y_temp,
            test_size=0.5,
            random_state=args.random_state,
            stratify=y_temp,
        )
        x_val, x_test, y_val, y_test = split_payload
        return x_train, x_val, x_test, y_train, y_val, y_test

    x_train_batches = []
    x_val_batches = []
    x_test_batches = []
    y_train_batches = []
    y_val_batches = []
    y_test_batches = []

    start = 0
    for item in stats:
        count = int(item["kept_window_count"])
        end = start + count
        x_label = x_audio[start:end]
        y_label = y[start:end]
        start = end

        train_count, val_count, test_count = split_count_triplet(count)
        train_end = train_count
        val_end = train_count + val_count

        x_train_batches.append(x_label[:train_end])
        x_val_batches.append(x_label[train_end:val_end])
        x_test_batches.append(x_label[val_end:val_end + test_count])
        y_train_batches.append(y_label[:train_end])
        y_val_batches.append(y_label[train_end:val_end])
        y_test_batches.append(y_label[val_end:val_end + test_count])

    return (
        np.concatenate(x_train_batches, axis=0),
        np.concatenate(x_val_batches, axis=0),
        np.concatenate(x_test_batches, axis=0),
        np.concatenate(y_train_batches, axis=0),
        np.concatenate(y_val_batches, axis=0),
        np.concatenate(y_test_batches, axis=0),
    )


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.random_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = select_labels(load_config(args.config), args.labels)
    x_audio, y, label_to_int, stats = build_time_aligned_audio_dataset(config, args)
    validate_class_counts(y, label_to_int)
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(
        x_audio,
        y,
        stats,
        args,
    )

    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight = {
        int(label): float(weight)
        for label, weight in zip(np.unique(y_train), class_weight_values)
    }

    model = build_audio_model(x_train.shape[1:], len(label_to_int))
    model.compile(
        optimizer=build_optimizer(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=100,
                restore_best_weights=True,
            )
        ],
    )

    test_metrics = model.evaluate(
        x_test,
        y_test,
        batch_size=args.batch_size,
        verbose=0,
        return_dict=True,
    )
    test_logits = model.predict(x_test, batch_size=args.batch_size, verbose=0)
    y_pred = np.argmax(test_logits, axis=1)
    class_names = class_names_from_map(label_to_int)
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    plot_history(
        history,
        args.output_dir / "audio_spectrogram_accuracy_loss.png",
        "Time-Aligned Audio Spectrogram Training Curves",
        ["accuracy", "loss"],
    )
    save_confusion_matrix_figure(
        y_test,
        y_pred,
        class_names,
        args.output_dir / "audio_spectrogram_confusion_matrix.png",
    )

    model.save(args.output_dir / "audio_spectrogram_cnn.keras")
    with (args.output_dir / "label_to_int.json").open("w", encoding="utf-8") as handle:
        json.dump(label_to_int, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "paired_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "dataset_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "audio_spectrogram_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": "time_aligned_audio_spectrogram",
                "target_sr": args.target_sr,
                "window_sec": args.window_sec,
                "hop_sec": args.hop_sec,
                "split_mode": args.split_mode,
                "highpass_hz": args.highpass_hz,
                "energy_quantile": args.energy_quantile,
                "max_windows_per_class": args.max_windows_per_class,
                "audio_offset_sec": args.audio_offset_sec,
                "vibration_offset_sec": args.vibration_offset_sec,
                "shared_duration_sec": args.shared_duration_sec,
                "nperseg": args.nperseg,
                "noverlap": args.noverlap,
                "max_freq_hz": args.max_freq_hz,
                "label_to_int": label_to_int,
                "model": "audio_spectrogram_cnn.keras",
                "input_shape": list(x_train.shape[1:]),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    with (args.output_dir / "audio_spectrogram_classification_report.txt").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write(report)

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    metrics_payload = {
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "window_sec": float(args.window_sec),
        "hop_sec": float(args.hop_sec),
        "support": int(len(y_test)),
    }
    for class_name in class_names:
        metrics_payload[f"{class_name}_precision"] = float(report_dict[class_name]["precision"])
        metrics_payload[f"{class_name}_recall"] = float(report_dict[class_name]["recall"])
        metrics_payload[f"{class_name}_f1"] = float(report_dict[class_name]["f1-score"])
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)

    print("Time-aligned audio spectrogram CNN test metrics:", test_metrics)
    print("Time-aligned audio spectrogram classification report:")
    print(report)


if __name__ == "__main__":
    main()
