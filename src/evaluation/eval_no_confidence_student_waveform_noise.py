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

from audio_student_model import (
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
    save_confusion_matrix_figure,
    select_labels,
)
from train_audio_student_distill_tcn_teacher import (
    DEFAULT_TIME_ALIGNED_TEACHER,
    build_grouped_dataset_for_tcn_teacher,
    class_names_from_map,
    load_tcn_teacher_model,
    resolve_sync_overrides,
    split_grouped_dataset,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_NO_CONFIDENCE_STUDENT = (
    "results/distill_outputs_paper_kl_ce_no_confidence_old/student_audio_only.keras"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an existing no-confidence audio student under waveform-level "
            "additive Gaussian noise."
        )
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--audio-target-sr", type=int, default=16000)
    parser.add_argument("--event-duration-sec", type=float, default=0.8)
    parser.add_argument("--hop-sec", type=float, default=0.8)
    parser.add_argument("--audio-highpass-hz", type=float, default=80.0)
    parser.add_argument("--nperseg", type=int, default=512)
    parser.add_argument("--noverlap", type=int, default=384)
    parser.add_argument("--max-freq-hz", type=float, default=6000.0)
    parser.add_argument("--audio-offset-sec", type=float, default=0.0)
    parser.add_argument("--vibration-offset-sec", type=float, default=0.0)
    parser.add_argument("--shared-duration-sec", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        choices=["blocked_time", "random"],
        default="blocked_time",
    )
    parser.add_argument(
        "--teacher-model",
        type=Path,
        default=DEFAULT_TIME_ALIGNED_TEACHER,
        help="Teacher checkpoint used only to infer teacher window size and reproduce paired windows.",
    )
    parser.add_argument(
        "--student-model",
        type=Path,
        default=DEFAULT_NO_CONFIDENCE_STUDENT,
        help="Existing no-confidence student_audio_only.keras checkpoint to evaluate.",
    )
    parser.add_argument(
        "--noise-snr-db-list",
        type=float,
        nargs="+",
        default=[30.0, 20.0, 10.0, 5.0, 0.0],
        help="SNR values in dB for waveform-level noisy test evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/distill_outputs_paper_kl_ce_no_confidence_waveform_noise_snr_test"),
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
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = student.evaluate(
        x_eval,
        y_eval,
        batch_size=batch_size,
        verbose=0,
        return_dict=True,
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
            paths,
            args,
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


def evaluate_waveform_noise_sweep(
    student,
    waveform_test,
    y_test,
    class_names,
    args,
):
    rows = []
    rng = np.random.default_rng(args.random_state)
    noise_root = args.output_dir / "waveform_noise_snr_eval"
    noise_root.mkdir(parents=True, exist_ok=True)

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
        print(f"Waveform-noisy no-confidence student SNR={snr_db:g} dB:")
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


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.student_model.exists():
        raise FileNotFoundError(f"Student model not found: {args.student_model}")
    if not args.teacher_model.exists():
        raise FileNotFoundError(f"Teacher model not found: {args.teacher_model}")

    paired_config = select_labels(load_config(args.config), args.labels)
    teacher_probe = load_tcn_teacher_model(args.teacher_model)
    teacher_window_size = int(teacher_probe.input_shape[1])

    x_audio, _teacher_window_groups, y, label_to_int, dataset_stats = build_grouped_dataset_for_tcn_teacher(
        paired_config,
        args,
        teacher_window_size=teacher_window_size,
    )
    (
        _x_audio_train,
        _x_audio_val,
        x_audio_test,
        _teacher_groups_train,
        _teacher_groups_val,
        _teacher_groups_test,
        _y_train,
        _y_val,
        y_test,
    ) = split_grouped_dataset(
        x_audio,
        _teacher_window_groups,
        y,
        dataset_stats,
        args,
    )

    student = keras.models.load_model(args.student_model, compile=False)
    expected_input_shape = tuple(x_audio_test.shape[1:])
    actual_input_shape = tuple(student.input_shape[1:])
    if actual_input_shape != expected_input_shape:
        raise ValueError(
            f"Student input shape {actual_input_shape} does not match expected {expected_input_shape}."
        )
    if int(student.output_shape[-1]) != len(label_to_int):
        raise ValueError(
            f"Student outputs {student.output_shape[-1]} classes, but this run expects {len(label_to_int)}."
        )
    student.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    class_names = class_names_from_map(label_to_int)
    clean_metrics, clean_report = evaluate_student_on_inputs(
        student,
        x_audio_test,
        y_test,
        class_names,
        args.output_dir,
        args.batch_size,
    )

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
        raise ValueError("Waveform noisy-test split does not match clean test split.")

    noise_rows = evaluate_waveform_noise_sweep(
        student,
        waveform_test,
        y_test,
        class_names,
        args,
    )

    with (args.output_dir / "noise_eval_run_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "variant": "no_confidence_existing_student_waveform_noise_eval",
                "student_model": str(args.student_model),
                "teacher_model": str(args.teacher_model),
                "teacher_window_size": int(teacher_window_size),
                "split_mode": args.split_mode,
                "audio_target_sr": args.audio_target_sr,
                "event_duration_sec": args.event_duration_sec,
                "hop_sec": args.hop_sec,
                "audio_highpass_hz": args.audio_highpass_hz,
                "nperseg": args.nperseg,
                "noverlap": args.noverlap,
                "max_freq_hz": args.max_freq_hz,
                "noise_snr_db_list": [float(value) for value in args.noise_snr_db_list],
                "noise_domain": "waveform",
                "clean_test_loss": float(clean_metrics["loss"]),
                "clean_test_accuracy": float(clean_metrics["accuracy"]),
                "noise_eval": noise_rows,
                "label_to_int": label_to_int,
                "student_input_shape": list(x_audio_test.shape[1:]),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("Clean no-confidence student classification report:")
    print(clean_report)


if __name__ == "__main__":
    main()
