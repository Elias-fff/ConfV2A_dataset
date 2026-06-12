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

from audio_student_model import build_optimizer, load_audio_signal
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a new TCN teacher from time-aligned vibration windows."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--audio-target-sr", type=int, default=16000)
    parser.add_argument("--event-duration-sec", type=float, default=0.8)
    parser.add_argument(
        "--hop-sec",
        type=float,
        default=0.3,
        help="Window hop in seconds. Smaller than event-duration-sec adds overlap.",
    )
    parser.add_argument("--audio-highpass-hz", type=float, default=80.0)
    parser.add_argument("--teacher-window-size", type=int, default=243)
    parser.add_argument("--audio-offset-sec", type=float, default=0.0)
    parser.add_argument("--vibration-offset-sec", type=float, default=0.0)
    parser.add_argument("--shared-duration-sec", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--class-weight",
        nargs="*",
        default=None,
        help=(
            "Optional manual class weights formatted as label=value, "
            "for example bottle_can_500=1.2 error=1.5."
        ),
    )
    parser.add_argument(
        "--split-mode",
        choices=["blocked_time", "random"],
        default="blocked_time",
        help="Use blocked_time to keep train/val/test ordered along each recording timeline.",
    )
    parser.add_argument(
        "--model-variant",
        choices=["base", "strong_v2"],
        default="strong_v2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("time_aligned_tcn_teacher_outputs"),
    )
    return parser.parse_args()


def class_names_from_map(label_to_int):
    return [label for label, _ in sorted(label_to_int.items(), key=lambda item: item[1])]


def build_base_tcn_teacher(input_shape, num_classes):
    look_back, dim = input_shape
    inputs = keras.Input((look_back, dim), name="vibration_input")
    x = keras.layers.Dense(32, activation="relu", name="teacher_embedding")(inputs)
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
    features = keras.layers.Dense(32, name="teacher_feature")(x)
    logits = keras.layers.Dense(num_classes, name="teacher_logits")(features)
    return keras.Model(inputs, logits, name="time_aligned_tcn_teacher")


def build_strong_vibration_teacher_v2(input_shape, num_classes):
    time_steps, _feature_dim = input_shape
    inputs = keras.Input(shape=input_shape, name="vibration_input")

    x = keras.layers.LayerNormalization(name="teacher_ln_in")(inputs)
    x = keras.layers.GaussianNoise(0.02, name="teacher_noise")(x)
    x = keras.layers.Conv1D(
        32, 5, padding="same", use_bias=False, name="teacher_conv1"
    )(x)
    x = keras.layers.BatchNormalization(name="teacher_bn1")(x)
    x = keras.layers.Activation("relu", name="teacher_relu1")(x)

    x = keras.layers.Conv1D(
        32, 3, padding="same", use_bias=False, name="teacher_conv2"
    )(x)
    x = keras.layers.BatchNormalization(name="teacher_bn2")(x)
    x = keras.layers.Activation("relu", name="teacher_relu2")(x)
    x = keras.layers.SpatialDropout1D(0.12, name="teacher_spatial_dropout")(x)

    x = keras.layers.Dense(32, activation="relu", name="teacher_embedding")(x)
    x = TCN(
        seq_len=time_steps,
        filters_list=[32, 64, 64],
        kernel_size_list=[3, 5, 5],
        name="teacher_tcn",
    )(x)

    avg_pool = keras.layers.GlobalAveragePooling1D(name="teacher_gap")(x)
    max_pool = keras.layers.GlobalMaxPooling1D(name="teacher_gmp")(x)
    x = keras.layers.Concatenate(name="teacher_pool_concat")([avg_pool, max_pool])

    x = keras.layers.Dense(96, activation="relu", name="teacher_dense1")(x)
    x = keras.layers.Dropout(0.3, name="teacher_dropout1")(x)
    features = keras.layers.Dense(32, name="teacher_feature")(x)
    logits = keras.layers.Dense(num_classes, name="teacher_logits")(features)
    return keras.Model(inputs=inputs, outputs=logits, name="strong_vibration_teacher_v2")


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


def build_time_aligned_teacher_dataset(config, args):
    x_batches = []
    y_batches = []
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
        numeric_frame, vibration_times = load_vibration_frame(vibration_path)
        audio_offset_sec, vibration_offset_sec, shared_duration_override = resolve_sync_overrides(
            paths, args
        )
        sync = synchronized_time_range(
            audio_num_samples=len(audio_signal),
            audio_sample_rate=args.audio_target_sr,
            vibration_time_seconds=vibration_times,
            audio_offset_sec=audio_offset_sec,
            vibration_offset_sec=vibration_offset_sec,
            shared_duration_sec=shared_duration_override,
        )

        aligned_audio = audio_signal[sync["audio_start_sample"] : sync["audio_end_sample"]]
        vibration_values, aligned_vibration_times = crop_vibration_to_time_range(
            numeric_frame,
            vibration_times,
            start_sec=sync["vibration_start_sec"],
            end_sec=sync["vibration_end_sec"],
        )
        vibration_values = zscore_normalize(vibration_values)

        if len(aligned_audio) < audio_window or len(vibration_values) == 0:
            continue

        start_samples = range(0, len(aligned_audio) - audio_window + 1, audio_hop)
        label_windows = []
        for start_sample in start_samples:
            start_time_sec = float(start_sample) / float(args.audio_target_sr)
            end_time_sec = start_time_sec + (audio_window / float(args.audio_target_sr))
            window = resample_vibration_interval(
                values=vibration_values,
                time_seconds=aligned_vibration_times,
                start_sec=start_time_sec,
                end_sec=end_time_sec,
                target_length=args.teacher_window_size,
            )
            if len(window) == 0:
                continue
            label_windows.append(window)

        if not label_windows:
            continue

        x_label = np.asarray(label_windows, dtype=np.float32)
        y_label = np.full((len(x_label),), label_to_int[label], dtype=np.int32)
        x_batches.append(x_label)
        y_batches.append(y_label)
        dataset_stats.append(
            {
                "label": label,
                "audio_path": str(audio_path),
                "vibration_path": str(vibration_path),
                "window_count": int(len(x_label)),
                "shared_duration_sec": float(sync["shared_duration_sec"]),
                "audio_duration_sec": float(sync["audio_duration_sec"]),
                "vibration_duration_sec": float(sync["vibration_duration_sec"]),
                "audio_offset_sec": float(sync["audio_offset_sec"]),
                "vibration_offset_sec": float(sync["vibration_offset_sec"]),
                "teacher_window_size": int(args.teacher_window_size),
                "event_duration_sec": float(args.event_duration_sec),
                "hop_sec": float(args.hop_sec),
            }
        )

    if not x_batches:
        raise ValueError("No time-aligned TCN windows were built. Check the paired config and sync settings.")

    x = np.concatenate(x_batches, axis=0)
    y = np.concatenate(y_batches, axis=0)
    return x, y, label_to_int, dataset_stats


def build_teacher_model(args, input_shape, num_classes):
    if args.model_variant == "strong_v2":
        return build_strong_vibration_teacher_v2(input_shape, num_classes)
    return build_base_tcn_teacher(input_shape, num_classes)


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


def split_dataset(x, y, stats, args):
    if args.split_mode == "random":
        split_payload = train_test_split(
            x,
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
        count = int(item["window_count"])
        end = start + count
        x_label = x[start:end]
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


def parse_manual_class_weights(raw_items, label_to_int):
    if not raw_items:
        return None

    class_weight = {
        class_index: 1.0 for class_index in label_to_int.values()
    }
    for item in raw_items:
        if "=" not in item:
            raise ValueError(
                f"Invalid class weight '{item}'. Expected format label=value."
            )
        label, value = item.split("=", 1)
        label = label.strip()
        if label not in label_to_int:
            available = ", ".join(sorted(label_to_int))
            raise ValueError(
                f"Unknown class weight label '{label}'. Available labels: {available}"
            )
        class_weight[label_to_int[label]] = float(value)
    return class_weight


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.random_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paired_config = select_labels(load_config(args.config), args.labels)
    x, y, label_to_int, dataset_stats = build_time_aligned_teacher_dataset(
        paired_config,
        args,
    )
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(
        x,
        y,
        dataset_stats,
        args,
    )
    class_weight = parse_manual_class_weights(args.class_weight, label_to_int)

    model = build_teacher_model(args, x_train.shape[1:], len(label_to_int))
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
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=300,
                restore_best_weights=True,
            )
        ],
        class_weight=class_weight,
    )

    plot_history(
        history,
        args.output_dir / "teacher_accuracy_loss.png",
        "Time-Aligned TCN Teacher Training Curves",
        ["accuracy", "loss"],
    )

    test_metrics = model.evaluate(
        x_test,
        y_test,
        batch_size=args.batch_size,
        verbose=0,
        return_dict=True,
    )
    logits = model.predict(x_test, batch_size=args.batch_size, verbose=0)
    y_pred = np.argmax(logits, axis=1)
    class_names = class_names_from_map(label_to_int)
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
        args.output_dir / "teacher_confusion_matrix.png",
    )

    model.save(args.output_dir / "teacher_vibration.keras")
    with (args.output_dir / "teacher_classification_report.txt").open("w", encoding="utf-8") as handle:
        handle.write(report)
    with (args.output_dir / "label_to_int.json").open("w", encoding="utf-8") as handle:
        json.dump(label_to_int, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "window_dataset_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_stats, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "teacher_run_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": "time_aligned_tcn_teacher",
                "model_variant": args.model_variant,
                "teacher_window_size": args.teacher_window_size,
                "audio_target_sr": args.audio_target_sr,
                "event_duration_sec": args.event_duration_sec,
                "hop_sec": args.hop_sec,
                "split_mode": args.split_mode,
                "audio_highpass_hz": args.audio_highpass_hz,
                "audio_offset_sec": args.audio_offset_sec,
                "vibration_offset_sec": args.vibration_offset_sec,
                "shared_duration_sec": args.shared_duration_sec,
                "test_loss": float(test_metrics["loss"]),
                "test_accuracy": float(test_metrics["accuracy"]),
                "class_weight": class_weight,
                "label_to_int": label_to_int,
                "input_shape": list(x_train.shape[1:]),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("Time-aligned TCN teacher classification report:")
    print(report)


if __name__ == "__main__":
    main()
