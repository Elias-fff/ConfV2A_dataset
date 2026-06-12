import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from audio_student_model import (
    compute_log_spectrograms,
    load_audio_signal,
    sliding_windows_1d,
)


DEFAULT_RUN_DIR = Path(
    "/Users/slade/Desktop/improve/distill_outputs_paper_kl_ce_noise"
)
DEFAULT_OUTPUT_DIR = Path("/Users/slade/Desktop/improve/test")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a distilled audio-only student model on a wav file and predict the bottle class."
        )
    )
    parser.add_argument("audio_path", type=Path, help="Input wav file to classify.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=(
            "Directory containing student_audio_only.keras, label_to_int.json, and "
            "distill_run_meta.json."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional direct path to a student .keras model. Overrides --run-dir model.",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="Optional direct path to label_to_int.json. Overrides --run-dir label map.",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=None,
        help="Optional direct path to distill_run_meta.json. Overrides --run-dir meta.",
    )
    parser.add_argument(
        "--aggregation",
        choices=["mean", "max"],
        default="mean",
        help="How to aggregate per-window probabilities into one file-level prediction.",
    )
    parser.add_argument(
        "--top-k-windows",
        type=int,
        default=5,
        help="How many highest-confidence windows to print.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save the inference result as JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory used to store result.json.",
    )
    return parser.parse_args()


def resolve_paths(args):
    run_dir = args.run_dir
    model_path = args.model_path or (run_dir / "student_audio_only.keras")
    label_map_path = args.label_map or (run_dir / "label_to_int.json")
    meta_path = args.meta_path or (run_dir / "distill_run_meta.json")

    missing = [str(path) for path in [model_path, label_map_path, meta_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required inference file(s): " + ", ".join(missing)
        )
    return model_path, label_map_path, meta_path


def load_runtime_config(meta_path):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "audio_target_sr": int(meta["audio_target_sr"]),
        "event_duration_sec": float(meta["event_duration_sec"]),
        "hop_sec": float(meta.get("student_hop_sec", meta.get("hop_sec", meta["event_duration_sec"]))),
        "audio_highpass_hz": float(meta["audio_highpass_hz"]),
        "nperseg": int(meta["nperseg"]),
        "noverlap": int(meta["noverlap"]),
        "max_freq_hz": float(meta["max_freq_hz"]),
        "student_input_shape": tuple(meta["student_input_shape"]),
    }


def build_windows(audio_signal, sample_rate, event_duration_sec, hop_sec):
    window_size = max(1, round(event_duration_sec * sample_rate))
    hop_size = max(1, round(hop_sec * sample_rate))
    windows = sliding_windows_1d(audio_signal, window_size, hop_size)
    start_times = (
        np.arange(len(windows), dtype=np.float32) * hop_size / float(sample_rate)
        if len(windows) > 0
        else np.empty((0,), dtype=np.float32)
    )
    return windows, start_times


def aggregate_probabilities(probabilities, mode):
    if mode == "max":
        return np.max(probabilities, axis=0)
    return np.mean(probabilities, axis=0)


def main():
    args = parse_args()
    model_path, label_map_path, meta_path = resolve_paths(args)
    runtime = load_runtime_config(meta_path)

    label_to_int = json.loads(label_map_path.read_text(encoding="utf-8"))
    int_to_label = {int(index): label for label, index in label_to_int.items()}

    _, audio_signal = load_audio_signal(
        args.audio_path,
        target_sr=runtime["audio_target_sr"],
        highpass_hz=runtime["audio_highpass_hz"],
    )
    windows, start_times = build_windows(
        audio_signal,
        sample_rate=runtime["audio_target_sr"],
        event_duration_sec=runtime["event_duration_sec"],
        hop_sec=runtime["hop_sec"],
    )
    if len(windows) == 0:
        raise ValueError(
            f"Audio is shorter than one inference window ({runtime['event_duration_sec']} s)."
        )

    specs = compute_log_spectrograms(
        windows,
        sample_rate=runtime["audio_target_sr"],
        nperseg=runtime["nperseg"],
        noverlap=runtime["noverlap"],
        max_freq_hz=runtime["max_freq_hz"],
    )
    if len(specs) == 0:
        raise ValueError("No spectrogram windows were produced from the input audio.")

    model = keras.models.load_model(model_path, compile=False)
    expected_shape = tuple(runtime["student_input_shape"])
    actual_shape = tuple(specs.shape[1:])
    if expected_shape != actual_shape:
        raise ValueError(
            f"Model expects input shape {expected_shape}, but extracted spectrograms are {actual_shape}."
        )

    logits = model.predict(specs, verbose=0)
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()
    aggregated = aggregate_probabilities(probabilities, args.aggregation)
    predicted_index = int(np.argmax(aggregated))
    predicted_label = int_to_label[predicted_index]

    per_window_pred = np.argmax(probabilities, axis=-1)
    per_window_conf = np.max(probabilities, axis=-1)
    top_k = max(0, min(args.top_k_windows, len(probabilities)))
    top_indices = np.argsort(-per_window_conf)[:top_k]

    result = {
        "audio_path": str(args.audio_path),
        "model_path": str(model_path),
        "aggregation": args.aggregation,
        "predicted_label": predicted_label,
        "predicted_index": predicted_index,
        "file_level_probabilities": {
            int_to_label[index]: float(aggregated[index]) for index in range(len(aggregated))
        },
        "window_count": int(len(specs)),
        "event_duration_sec": runtime["event_duration_sec"],
        "hop_sec": runtime["hop_sec"],
        "top_windows": [
            {
                "window_index": int(index),
                "start_sec": float(start_times[index]),
                "predicted_label": int_to_label[int(per_window_pred[index])],
                "confidence": float(per_window_conf[index]),
                "probabilities": {
                    int_to_label[class_index]: float(probabilities[index, class_index])
                    for class_index in range(probabilities.shape[1])
                },
            }
            for index in top_indices
        ],
    }

    print(f"Predicted label: {predicted_label}")
    print("File-level probabilities:")
    for label, prob in sorted(
        result["file_level_probabilities"].items(), key=lambda item: item[1], reverse=True
    ):
        print(f"  {label}: {prob:.4f}")

    if top_k > 0:
        print(f"\nTop {top_k} window predictions:")
        for item in result["top_windows"]:
            print(
                f"  window={item['window_index']:>3} "
                f"start={item['start_sec']:.2f}s "
                f"label={item['predicted_label']} "
                f"conf={item['confidence']:.4f}"
            )

    save_json_path = args.save_json
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_json_path = args.output_dir / "result.json"

    if save_json_path is not None:
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
        save_json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved result to: {save_json_path}")


if __name__ == "__main__":
    main()
