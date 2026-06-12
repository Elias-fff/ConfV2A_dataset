import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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


DEFAULT_AUDIO_ONLY_CONFIG = {
    "bottle_1000": "sensor_plot_project/bottle_1/1.wav",
    "bottle_500": "sensor_plot_project/bottle_500/500ml.wav",
    "bottle_can_330": "sensor_plot_project/bottle_yi_330/yi_330.wav",
    "bottle_can_500": "sensor_plot_project/bottle_yi_500/yi_500.wav",
    "error": "sensor_plot_project/error/error.wav",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a pure-audio spectrogram CNN classifier."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON mapping class names to wav paths or {'audio': path}.",
    )
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--window-sec", type=float, default=1.5)
    parser.add_argument(
        "--window-sec-list",
        type=float,
        nargs="+",
        default=None,
        help="Optional sweep over multiple window sizes in seconds.",
    )
    parser.add_argument("--hop-sec", type=float, default=0.3)
    parser.add_argument(
        "--highpass-hz",
        type=float,
        default=80.0,
        help="Set to 0 to disable the high-pass denoise step.",
    )
    parser.add_argument(
        "--energy-quantile",
        type=float,
        default=0.35,
        help="Drop the lowest-energy windows below this quantile.",
    )
    parser.add_argument(
        "--max-windows-per-class",
        type=int,
        default=500,
        help="Optional cap after filtering to keep the classes balanced.",
    )
    parser.add_argument("--nperseg", type=int, default=512)
    parser.add_argument("--noverlap", type=int, default=384)
    parser.add_argument("--max-freq-hz", type=float, default=6000.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--seed-list",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of random seeds to repeat each window size.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("audio_spectrogram_outputs"),
        help="Directory used to store the model and figures.",
    )
    return parser.parse_args()


def load_config(path):
    if path is None:
        return DEFAULT_AUDIO_ONLY_CONFIG
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_audio_path(entry):
    if isinstance(entry, str):
        return Path(entry)
    if isinstance(entry, dict) and "audio" in entry:
        return Path(entry["audio"])
    raise ValueError(f"Unsupported config entry: {entry!r}")


def build_audio_dataset(config, args):
    rng = np.random.default_rng(args.random_state)
    class_windows = []
    labels = []
    label_to_int = {label: index for index, label in enumerate(sorted(config))}
    stats = []

    window_size = max(1, round(args.window_sec * args.target_sr))
    hop_size = max(1, round(args.hop_sec * args.target_sr))

    for label, entry in config.items():
        audio_path = resolve_audio_path(entry)
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing audio file: {audio_path}")

        sample_rate, audio_signal = load_audio_signal(
            audio_path,
            target_sr=args.target_sr,
            highpass_hz=args.highpass_hz,
        )
        raw_windows = sliding_windows_1d(audio_signal, window_size, hop_size)
        kept_windows, kept_rms = filter_windows_by_energy(
            raw_windows,
            energy_quantile=args.energy_quantile,
            max_windows=args.max_windows_per_class,
            rng=rng,
        )
        if len(kept_windows) == 0:
            continue

        specs = compute_log_spectrograms(
            kept_windows,
            sample_rate=sample_rate,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
            max_freq_hz=args.max_freq_hz,
        )
        class_windows.append(specs)
        labels.append(np.full((len(specs),), label_to_int[label], dtype=np.int32))
        stats.append(
            {
                "label": label,
                "audio_path": str(audio_path),
                "raw_window_count": int(len(raw_windows)),
                "kept_window_count": int(len(specs)),
                "mean_rms": float(np.mean(kept_rms)),
            }
        )

    if not class_windows:
        raise ValueError("No audio samples were built. Check the config paths and window settings.")

    x_audio = np.concatenate(class_windows, axis=0)
    y = np.concatenate(labels, axis=0)
    return x_audio, y, label_to_int, stats


def build_audio_model(input_shape, num_classes):
    return build_audio_student_model(
        input_shape=input_shape,
        num_classes=num_classes,
        input_name="spectrogram_input",
        model_name="audio_spectrogram_cnn",
        prefix=None,
    )


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
            f"Current counts: {readable}. Increase the audio duration coverage, reduce "
            "--energy-quantile, or raise --max-windows-per-class."
        )

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
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=color)

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


def train_one_run(args, output_dir):
    keras.utils.set_random_seed(args.random_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    x_audio, y, label_to_int, stats = build_audio_dataset(config, args)
    validate_class_counts(y, label_to_int)

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
                patience=10,
                restore_best_weights=True,
            )
        ],
    )

    test_metrics = model.evaluate(
        x_test, y_test, batch_size=args.batch_size, verbose=0, return_dict=True
    )
    print("Audio spectrogram CNN test metrics:", test_metrics)

    test_logits = model.predict(x_test, batch_size=args.batch_size, verbose=0)
    y_pred = np.argmax(test_logits, axis=1)
    class_names = [
        label for label, _ in sorted(label_to_int.items(), key=lambda item: item[1])
    ]
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print("Audio spectrogram CNN classification report:")
    print(report)

    plot_history(
        history,
        output_dir / "audio_spectrogram_accuracy_loss.png",
        "Audio Spectrogram CNN Training Curves",
        ["accuracy", "loss"],
    )
    save_confusion_matrix_figure(
        y_test,
        y_pred,
        class_names,
        output_dir / "audio_spectrogram_confusion_matrix.png",
    )

    model.save(output_dir / "audio_spectrogram_cnn.keras")
    with (output_dir / "label_to_int.json").open("w", encoding="utf-8") as handle:
        json.dump(label_to_int, handle, ensure_ascii=False, indent=2)
    with (output_dir / "audio_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    with (output_dir / "dataset_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    with (output_dir / "audio_spectrogram_meta.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "target_sr": args.target_sr,
                "window_sec": args.window_sec,
                "hop_sec": args.hop_sec,
                "highpass_hz": args.highpass_hz,
                "energy_quantile": args.energy_quantile,
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
    with (output_dir / "audio_spectrogram_classification_report.txt").open(
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
        "random_state": int(args.random_state),
        "support": int(len(y_test)),
    }
    for class_name in class_names:
        metrics_payload[f"{class_name}_precision"] = float(report_dict[class_name]["precision"])
        metrics_payload[f"{class_name}_recall"] = float(report_dict[class_name]["recall"])
        metrics_payload[f"{class_name}_f1"] = float(report_dict[class_name]["f1-score"])
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)
    return metrics_payload


def summarize_sweep(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record["window_sec"], []).append(record)

    summary_rows = []
    for window_sec in sorted(grouped):
        rows = grouped[window_sec]
        accuracy_values = np.asarray([row["accuracy"] for row in rows], dtype=np.float32)
        macro_f1_values = np.asarray([row["macro_f1"] for row in rows], dtype=np.float32)
        weighted_f1_values = np.asarray([row["weighted_f1"] for row in rows], dtype=np.float32)
        bottle_330_recall = np.asarray(
            [row.get("bottle_can_330_recall", 0.0) for row in rows], dtype=np.float32
        )
        bottle_500_recall = np.asarray(
            [row.get("bottle_can_500_recall", 0.0) for row in rows], dtype=np.float32
        )
        summary_rows.append(
            {
                "window_sec": float(window_sec),
                "num_runs": int(len(rows)),
                "accuracy_mean": float(np.mean(accuracy_values)),
                "accuracy_std": float(np.std(accuracy_values)),
                "macro_f1_mean": float(np.mean(macro_f1_values)),
                "macro_f1_std": float(np.std(macro_f1_values)),
                "weighted_f1_mean": float(np.mean(weighted_f1_values)),
                "weighted_f1_std": float(np.std(weighted_f1_values)),
                "bottle_can_330_recall_mean": float(np.mean(bottle_330_recall)),
                "bottle_can_330_recall_std": float(np.std(bottle_330_recall)),
                "bottle_can_500_recall_mean": float(np.mean(bottle_500_recall)),
                "bottle_can_500_recall_std": float(np.std(bottle_500_recall)),
            }
        )
    return summary_rows


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(path, summary_rows):
    headers = [
        "Metric",
        *[f"{row['window_sec']:.1f}s" for row in summary_rows],
    ]
    metric_specs = [
        ("Accuracy (Mean)", "accuracy_mean"),
        ("Accuracy (Std)", "accuracy_std"),
        ("Macro F1 (Mean)", "macro_f1_mean"),
        ("Macro F1 (Std)", "macro_f1_std"),
        ("330 Recall (Mean)", "bottle_can_330_recall_mean"),
        ("330 Recall (Std)", "bottle_can_330_recall_std"),
        ("500 Recall (Mean)", "bottle_can_500_recall_mean"),
        ("500 Recall (Std)", "bottle_can_500_recall_std"),
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for label, key in metric_specs:
        values = [f"{row[key]:.4f}" for row in summary_rows]
        lines.append("| " + " | ".join([label, *values]) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    window_sec_list = args.window_sec_list if args.window_sec_list is not None else [args.window_sec]
    seed_list = args.seed_list if args.seed_list is not None else [args.random_state]

    detailed_rows = []
    for window_sec in window_sec_list:
        for seed in seed_list:
            run_args = argparse.Namespace(**vars(args))
            run_args.window_sec = float(window_sec)
            run_args.random_state = int(seed)
            run_dir = args.output_dir / f"window_{window_sec:.2f}".replace(".", "_") / f"seed_{seed}"
            print(f"\n=== Running window_sec={window_sec:.2f}, seed={seed} ===")
            metrics_payload = train_one_run(run_args, run_dir)
            detailed_row = {
                "window_sec": float(window_sec),
                "seed": int(seed),
                **metrics_payload,
            }
            detailed_rows.append(detailed_row)

    summary_rows = summarize_sweep(detailed_rows)
    write_csv(args.output_dir / "window_sweep_detailed.csv", detailed_rows)
    write_csv(args.output_dir / "window_sweep_summary.csv", summary_rows)
    write_markdown_summary(args.output_dir / "window_sweep_summary.md", summary_rows)
    with (args.output_dir / "window_sweep_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, ensure_ascii=False, indent=2)
    print("\nWindow sweep summary:")
    for row in summary_rows:
        print(
            f"window={row['window_sec']:.2f}s "
            f"acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f} "
            f"macro_f1={row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f} "
            f"330_recall={row['bottle_can_330_recall_mean']:.4f}"
        )


def main():
    args = parse_args()
    if args.window_sec_list is not None or args.seed_list is not None:
        run_sweep(args)
        return
    train_one_run(args, args.output_dir)


if __name__ == "__main__":
    main()

# 单次训练 python3 train_audio_spectrogram.py --window-sec 0.8 --output-dir audio_spectrogram_outputs_best
# 多次训练 生成表python3 train_audio_spectrogram.py \
  #--window-sec-list 0.5 0.8 1.2 1.5 \
  #--seed-list 42 \
  #--output-dir audio_spectrogram_window_sweep

#多次随机不同种子训练 python3 train_audio_spectrogram.py \
  #--window-sec-list 0.5 0.8 1.2 1.5 \
  #--seed-list 42 52 62 \
  #--output-dir audio_spectrogram_window_sweep
