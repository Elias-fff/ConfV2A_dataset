from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import find_peaks, resample_poly


def zscore_normalize(values):
    values = values.astype(np.float32)
    if values.size == 0:
        return values
    return (values - values.mean()) / (values.std() + 1e-8)


def load_audio_signal(path):
    sample_rate, signal = wavfile.read(path)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    return sample_rate, zscore_normalize(signal)


def resample_audio(signal, original_sr, target_sr):
    if target_sr == original_sr:
        return signal.astype(np.float32)
    return resample_poly(signal, target_sr, original_sr).astype(np.float32)


def load_vibration_frame(path):
    frame = pd.read_csv(path)
    if "timestamp" in frame.columns:
        time_seconds = pd.to_timedelta(frame["timestamp"]).dt.total_seconds().to_numpy()
        time_seconds = time_seconds - time_seconds[0]
        numeric_frame = frame.drop(columns=["timestamp"])
    else:
        time_seconds = np.arange(len(frame), dtype=np.float32)
        numeric_frame = frame
    return numeric_frame, time_seconds.astype(np.float32)


def unique_time_axis(time_seconds):
    time_seconds = np.asarray(time_seconds, dtype=np.float32)
    if len(time_seconds) == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    # CSV logs can contain duplicate timestamps, which break interpolation.
    last_positions = np.flatnonzero(np.r_[np.diff(time_seconds) > 0.0, True])
    unique_times = time_seconds[last_positions]
    unique_positions = last_positions.astype(np.float32)
    return unique_times, unique_positions


def map_times_to_sample_positions(time_seconds, query_times_sec):
    query_times_sec = np.asarray(query_times_sec, dtype=np.float32)
    if len(query_times_sec) == 0:
        return np.empty((0,), dtype=np.float32)

    unique_times, unique_positions = unique_time_axis(time_seconds)
    if len(unique_times) == 0:
        return np.empty((0,), dtype=np.float32)
    if len(unique_times) == 1:
        return np.full(query_times_sec.shape, unique_positions[0], dtype=np.float32)

    clipped_times = np.clip(query_times_sec, unique_times[0], unique_times[-1])
    return np.interp(clipped_times, unique_times, unique_positions).astype(np.float32)


def synchronized_time_range(
    audio_num_samples,
    audio_sample_rate,
    vibration_time_seconds,
    audio_offset_sec=0.0,
    vibration_offset_sec=0.0,
    shared_duration_sec=None,
):
    audio_duration_sec = float(audio_num_samples) / float(audio_sample_rate)
    vibration_duration_sec = float(vibration_time_seconds[-1]) if len(vibration_time_seconds) else 0.0

    audio_offset_sec = max(0.0, float(audio_offset_sec))
    vibration_offset_sec = max(0.0, float(vibration_offset_sec))

    available_audio_sec = max(0.0, audio_duration_sec - audio_offset_sec)
    available_vibration_sec = max(0.0, vibration_duration_sec - vibration_offset_sec)
    auto_shared_duration_sec = min(available_audio_sec, available_vibration_sec)

    if shared_duration_sec is None:
        resolved_shared_duration_sec = auto_shared_duration_sec
    else:
        resolved_shared_duration_sec = min(
            max(0.0, float(shared_duration_sec)),
            auto_shared_duration_sec,
        )

    audio_start_sample = min(
        int(round(audio_offset_sec * float(audio_sample_rate))),
        max(0, int(audio_num_samples)),
    )
    audio_shared_samples = min(
        int(round(resolved_shared_duration_sec * float(audio_sample_rate))),
        max(0, int(audio_num_samples) - audio_start_sample),
    )
    audio_end_sample = audio_start_sample + audio_shared_samples

    return {
        "audio_duration_sec": audio_duration_sec,
        "vibration_duration_sec": vibration_duration_sec,
        "audio_offset_sec": audio_offset_sec,
        "vibration_offset_sec": vibration_offset_sec,
        "shared_duration_sec": resolved_shared_duration_sec,
        "audio_start_sample": audio_start_sample,
        "audio_end_sample": audio_end_sample,
        "audio_shared_samples": audio_shared_samples,
        "vibration_start_sec": vibration_offset_sec,
        "vibration_end_sec": vibration_offset_sec + resolved_shared_duration_sec,
    }


def crop_vibration_to_time_range(numeric_frame, time_seconds, start_sec, end_sec):
    values = numeric_frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    if len(values) == 0 or len(time_seconds) == 0:
        feature_dim = values.shape[-1] if values.ndim == 2 else 0
        return (
            np.empty((0, feature_dim), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    mask = (time_seconds >= float(start_sec)) & (time_seconds <= float(end_sec))
    cropped_values = values[mask]
    cropped_times = time_seconds[mask].astype(np.float32) - float(start_sec)
    return cropped_values.astype(np.float32), cropped_times.astype(np.float32)


def resample_vibration_interval(values, time_seconds, start_sec, end_sec, target_length):
    values = np.asarray(values, dtype=np.float32)
    time_seconds = np.asarray(time_seconds, dtype=np.float32)
    target_length = int(target_length)

    feature_dim = values.shape[-1] if values.ndim == 2 else 0
    if len(values) == 0 or len(time_seconds) == 0 or target_length <= 0:
        return np.empty((0, feature_dim), dtype=np.float32)

    unique_times, unique_positions = unique_time_axis(time_seconds)
    if len(unique_times) == 0:
        return np.empty((0, feature_dim), dtype=np.float32)
    unique_values = values[unique_positions.astype(np.int32)]

    start_sec = float(np.clip(start_sec, unique_times[0], unique_times[-1]))
    end_sec = float(np.clip(end_sec, start_sec, unique_times[-1]))

    if target_length == 1:
        target_times = np.array([(start_sec + end_sec) * 0.5], dtype=np.float32)
    elif end_sec <= start_sec:
        target_times = np.full((target_length,), start_sec, dtype=np.float32)
    else:
        target_times = np.linspace(
            start_sec,
            end_sec,
            num=target_length,
            endpoint=False,
            dtype=np.float32,
        )

    resampled = [
        np.interp(target_times, unique_times, unique_values[:, dim]).astype(np.float32)
        for dim in range(unique_values.shape[1])
    ]
    return np.stack(resampled, axis=-1)


def vibration_sample_rate(time_seconds):
    if len(time_seconds) < 2:
        return 1.0
    diffs = np.diff(time_seconds)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1.0
    return float(1.0 / np.median(diffs))


def build_peak_signal(numeric_frame):
    if {"AN1", "AN2"}.issubset(numeric_frame.columns):
        signal = np.maximum(
            numeric_frame["AN1"].to_numpy(dtype=np.float32),
            numeric_frame["AN2"].to_numpy(dtype=np.float32),
        )
    else:
        signal = np.linalg.norm(numeric_frame.to_numpy(dtype=np.float32), axis=1)
    return zscore_normalize(signal)


def smooth_signal(signal, kernel_size):
    kernel_size = max(1, int(kernel_size))
    if kernel_size == 1:
        return signal
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(signal, kernel, mode="same")


def detect_event_peaks(
    vibration_csv,
    min_peak_distance_sec=1.8,
    prominence=1.0,
    smooth_sec=0.08,
):
    numeric_frame, time_seconds = load_vibration_frame(vibration_csv)
    peak_signal = build_peak_signal(numeric_frame)
    vib_rate = vibration_sample_rate(time_seconds)
    smooth_size = max(1, round(vib_rate * smooth_sec))
    peak_signal = smooth_signal(peak_signal, smooth_size)
    distance = max(1, round(vib_rate * min_peak_distance_sec))
    peaks, _ = find_peaks(peak_signal, distance=distance, prominence=prominence)
    peak_times = time_seconds[peaks] if len(peaks) else np.empty((0,), dtype=np.float32)
    vib_duration = float(time_seconds[-1]) if len(time_seconds) else 0.0
    return peaks, peak_times, vib_rate, vib_duration


def extract_windows_from_centers(signal, center_indices, window_length):
    half_left = window_length // 2
    half_right = window_length - half_left
    windows = []
    kept_positions = []
    for pos, center in enumerate(center_indices):
        start = int(center) - half_left
        end = int(center) + half_right
        if start < 0 or end > len(signal):
            continue
        windows.append(signal[start:end])
        kept_positions.append(pos)
    if not windows:
        shape = (0, window_length) if signal.ndim == 1 else (0, window_length, signal.shape[-1])
        return np.empty(shape, dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.asarray(windows, dtype=np.float32), np.asarray(kept_positions, dtype=np.int32)


def paired_event_windows(
    audio_path,
    vibration_path,
    audio_target_sr=4000,
    event_duration_sec=1.2,
    min_peak_distance_sec=1.8,
    prominence=1.0,
    smooth_sec=0.08,
):
    peaks, peak_times, vib_rate, vib_duration = detect_event_peaks(
        vibration_path,
        min_peak_distance_sec=min_peak_distance_sec,
        prominence=prominence,
        smooth_sec=smooth_sec,
    )
    numeric_frame, vibration_times = load_vibration_frame(vibration_path)
    audio_sr, audio_signal = load_audio_signal(audio_path)
    audio_signal = resample_audio(audio_signal, audio_sr, audio_target_sr)

    sync = synchronized_time_range(
        audio_num_samples=len(audio_signal),
        audio_sample_rate=audio_target_sr,
        vibration_time_seconds=vibration_times,
    )
    audio_signal = audio_signal[sync["audio_start_sample"] : sync["audio_end_sample"]]
    vibration_values, aligned_vibration_times = crop_vibration_to_time_range(
        numeric_frame,
        vibration_times,
        start_sec=sync["vibration_start_sec"],
        end_sec=sync["vibration_end_sec"],
    )
    if len(vibration_values) == 0:
        vibration_window_len = max(1, round(event_duration_sec * max(vib_rate, 1.0)))
        audio_window_len = max(1, round(event_duration_sec * audio_target_sr))
        feature_dim = vibration_values.shape[-1] if vibration_values.ndim == 2 else 0
        return (
            np.empty((0, audio_window_len, 1), dtype=np.float32),
            np.empty((0, vibration_window_len, feature_dim), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    vibration_signal = zscore_normalize(vibration_values)

    if len(audio_signal) == 0 or len(vibration_signal) == 0:
        vibration_window_len = max(1, round(event_duration_sec * max(vib_rate, 1.0)))
        audio_window_len = max(1, round(event_duration_sec * audio_target_sr))
        feature_dim = vibration_signal.shape[-1] if vibration_signal.ndim == 2 else 0
        return (
            np.empty((0, audio_window_len, 1), dtype=np.float32),
            np.empty((0, vibration_window_len, feature_dim), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    aligned_peak_times = peak_times - sync["vibration_start_sec"]
    valid_peak_mask = (
        (aligned_peak_times >= 0.0)
        & (aligned_peak_times <= sync["shared_duration_sec"])
    )
    aligned_peak_times = aligned_peak_times[valid_peak_mask]

    if len(aligned_peak_times) == 0:
        vibration_window_len = max(1, round(event_duration_sec * max(vib_rate, 1.0)))
        audio_window_len = max(1, round(event_duration_sec * audio_target_sr))
        return (
            np.empty((0, audio_window_len, 1), dtype=np.float32),
            np.empty((0, vibration_window_len, vibration_signal.shape[-1]), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    vibration_positions = np.round(
        map_times_to_sample_positions(aligned_vibration_times, aligned_peak_times)
    ).astype(np.int32)
    vibration_window_len = max(1, round(event_duration_sec * vib_rate))
    vibration_windows, kept_peak_positions = extract_windows_from_centers(
        vibration_signal, vibration_positions, vibration_window_len
    )
    if len(vibration_windows) == 0:
        return (
            np.empty((0, 1, 1), dtype=np.float32),
            np.empty((0, 1, vibration_signal.shape[-1]), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    scaled_peak_times = aligned_peak_times[kept_peak_positions]
    audio_centers = np.round(scaled_peak_times * audio_target_sr).astype(np.int32)
    audio_window_len = max(1, round(event_duration_sec * audio_target_sr))
    audio_windows, kept_audio_positions = extract_windows_from_centers(
        audio_signal, audio_centers, audio_window_len
    )
    if len(audio_windows) == 0:
        return (
            np.empty((0, audio_window_len, 1), dtype=np.float32),
            np.empty((0, vibration_window_len, vibration_signal.shape[-1]), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    vibration_windows = vibration_windows[kept_audio_positions]
    scaled_peak_times = scaled_peak_times[kept_audio_positions]
    paired_count = min(len(audio_windows), len(vibration_windows))
    return (
        audio_windows[:paired_count, ..., np.newaxis],
        vibration_windows[:paired_count],
        scaled_peak_times[:paired_count],
    )
