from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile
from tensorflow import keras


def normalize_waveform(data):
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = max(abs(info.min), info.max)
        data = data.astype(np.float32) / float(scale)
    else:
        data = data.astype(np.float32)
    return data


def load_audio_signal(path, target_sr, highpass_hz):
    sample_rate, signal_data = wavfile.read(Path(path))
    signal_data = normalize_waveform(signal_data)
    if signal_data.ndim == 2:
        signal_data = signal_data.mean(axis=1)
    signal_data = signal_data - np.mean(signal_data)

    if sample_rate != target_sr:
        signal_data = signal.resample_poly(signal_data, target_sr, sample_rate)
        sample_rate = target_sr

    if highpass_hz > 0.0 and highpass_hz < sample_rate / 2.0:
        sos = signal.butter(4, highpass_hz, btype="highpass", fs=sample_rate, output="sos")
        signal_data = signal.sosfiltfilt(sos, signal_data)

    signal_data = signal_data.astype(np.float32)
    signal_data = signal_data - np.mean(signal_data)
    signal_data = signal_data / (np.std(signal_data) + 1e-8)
    return sample_rate, signal_data


def sliding_windows_1d(signal_data, window_size, hop_size):
    if len(signal_data) < window_size:
        return np.empty((0, window_size), dtype=np.float32)

    windows = []
    for start in range(0, len(signal_data) - window_size + 1, hop_size):
        windows.append(signal_data[start : start + window_size])
    return np.asarray(windows, dtype=np.float32)


def filter_windows_by_energy(windows, energy_quantile, max_windows, rng):
    if len(windows) == 0:
        return windows, np.empty((0,), dtype=np.float32)

    rms = np.sqrt(np.mean(np.square(windows), axis=1))
    energy_quantile = float(np.clip(energy_quantile, 0.0, 0.95))
    threshold = np.quantile(rms, energy_quantile)
    keep_mask = rms >= threshold
    kept_indices = np.flatnonzero(keep_mask)

    if len(kept_indices) == 0:
        kept_indices = np.array([int(np.argmax(rms))], dtype=np.int32)

    if max_windows is not None and max_windows > 0 and len(kept_indices) > max_windows:
        kept_indices = np.sort(rng.choice(kept_indices, size=max_windows, replace=False))

    return windows[kept_indices], rms[kept_indices]


def compute_log_spectrograms(windows, sample_rate, nperseg, noverlap, max_freq_hz):
    spectrograms = []
    kept_freqs = None

    for window in windows:
        freqs, _, spec = signal.spectrogram(
            window,
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            scaling="spectrum",
            mode="magnitude",
        )
        if kept_freqs is None:
            if max_freq_hz > 0:
                kept_freqs = freqs <= max_freq_hz
            else:
                kept_freqs = np.ones_like(freqs, dtype=bool)
        spec = spec[kept_freqs]
        spec = np.log1p(spec)
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)
        spectrograms.append(spec[..., np.newaxis].astype(np.float32))

    if not spectrograms:
        freq_bins = int(np.sum(kept_freqs)) if kept_freqs is not None else 0
        return np.empty((0, freq_bins, 0, 1), dtype=np.float32)

    return np.asarray(spectrograms, dtype=np.float32)


def build_audio_student_model(input_shape, num_classes, input_name="spectrogram_input", model_name="audio_spectrogram_cnn", prefix=None):
    def layer_name(base):
        return f"{prefix}_{base}" if prefix else None

    inputs = keras.Input(shape=input_shape, name=input_name)
    x = keras.layers.Conv2D(
        16, 3, padding="same", use_bias=False, name=layer_name("conv1")
    )(inputs)
    x = keras.layers.BatchNormalization(name=layer_name("bn1"))(x)
    x = keras.layers.Activation("relu", name=layer_name("relu1"))(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name=layer_name("pool1"))(x)

    x = keras.layers.Conv2D(
        32, 3, padding="same", use_bias=False, name=layer_name("conv2")
    )(x)
    x = keras.layers.BatchNormalization(name=layer_name("bn2"))(x)
    x = keras.layers.Activation("relu", name=layer_name("relu2"))(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name=layer_name("pool2"))(x)

    x = keras.layers.Conv2D(
        64, 3, padding="same", use_bias=False, name=layer_name("conv3")
    )(x)
    x = keras.layers.BatchNormalization(name=layer_name("bn3"))(x)
    x = keras.layers.Activation("relu", name=layer_name("relu3"))(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name=layer_name("pool3"))(x)
    x = keras.layers.SpatialDropout2D(0.2, name=layer_name("spatial_dropout"))(x)

    x = keras.layers.Conv2D(
        96, 3, padding="same", use_bias=False, name=layer_name("conv4")
    )(x)
    x = keras.layers.BatchNormalization(name=layer_name("bn4"))(x)
    x = keras.layers.Activation("relu", name=layer_name("relu4"))(x)
    x = keras.layers.GlobalAveragePooling2D(name=layer_name("gap"))(x)
    x = keras.layers.Dense(96, activation="relu", name=layer_name("dense"))(x)
    x = keras.layers.Dropout(0.3, name=layer_name("dropout"))(x)
    feature_name = f"{prefix}_feature" if prefix else "audio_feature"
    logits_name = f"{prefix}_logits" if prefix else "audio_logits"
    features = keras.layers.Dense(32, name=feature_name)(x)
    logits = keras.layers.Dense(num_classes, name=logits_name)(features)
    return keras.Model(inputs=inputs, outputs=logits, name=model_name)


def build_optimizer(learning_rate=1e-3):
    legacy = getattr(keras.optimizers, "legacy", None)
    if legacy is not None and hasattr(legacy, "Adam"):
        return legacy.Adam(learning_rate=learning_rate)
    return keras.optimizers.Adam(learning_rate=learning_rate)
