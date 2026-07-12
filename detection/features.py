"""Handcrafted feature engineering for the detection pipeline's XGBoost
branch.

Computes a fixed-length feature vector per window (time-domain
per-channel statistics, cross-channel magnitude/jerk/tilt features, and
simple frequency-domain features), matching the kind of feature set
standard in IMU-based fall-detection literature (including SisFall's
and KFall's own papers) rather than anything exotic.

This module is intentionally decoupled from any specific model --
`compute_features_batch` just returns a plain DataFrame of numbers plus
label/provenance columns. `train_xgboost.py` is what turns that into
train/val/test splits and an actual trained model.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from detection.dataset import CHANNELS, load_window
from detection.windowing import WindowingConfig

# Per-channel time-domain stats computed for every one of the 6 raw
# channels (acc_x/y/z, gyro_x/y/z) -- 6 stats x 6 channels = 36 features.
_CHANNEL_STAT_NAMES = ["mean", "std", "min", "max", "range", "rms"]


def _channel_stats(x: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_range": float(np.max(x) - np.min(x)),
        f"{prefix}_rms": float(np.sqrt(np.mean(x ** 2))),
    }


def _power_spectrum(x: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Real FFT power spectrum of `x`, DC component removed (mean
    subtracted before transforming) so the spectrum reflects genuine
    oscillation/transient content, not the window's baseline offset.
    """
    n = len(x)
    if n < 2:
        return np.array([]), np.array([])
    fft_vals = np.fft.rfft(x - np.mean(x))
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    return freqs, power


def compute_window_features(window: np.ndarray, sample_rate_hz: float = 100.0) -> dict[str, float]:
    """Compute the full handcrafted feature vector for one window.

    `window` must be shape (n_samples, 6) in CHANNELS order
    (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z) -- exactly what
    `detection.dataset.load_window` returns.
    """
    if window.shape[1] != len(CHANNELS):
        raise ValueError(
            f"Expected {len(CHANNELS)} channels ({CHANNELS}), got shape {window.shape}"
        )

    features: dict[str, float] = {}

    # 1. Per-channel time-domain stats (36 features).
    for i, channel_name in enumerate(CHANNELS):
        features.update(_channel_stats(window[:, i], channel_name))

    acc = window[:, :3]
    gyro = window[:, 3:]
    accel_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    # 2. Acceleration/gyro magnitude stats (7 features) -- these
    # capture overall movement intensity independent of device
    # orientation, unlike the raw per-axis stats above.
    features.update(_channel_stats(accel_mag, "accel_mag"))
    features["gyro_mag_mean"] = float(np.mean(gyro_mag))
    features["gyro_mag_std"] = float(np.std(gyro_mag))
    features["gyro_mag_max"] = float(np.max(gyro_mag))

    # 3. Jerk (rate of change of acceleration magnitude) -- a sudden
    # spike in jerk is one of the strongest classical fall indicators
    # (the abrupt deceleration on impact), distinct from just "high
    # acceleration" which strenuous ADLs can also produce.
    jerk = np.diff(accel_mag) * sample_rate_hz if len(accel_mag) > 1 else np.array([0.0])
    features["jerk_mean_abs"] = float(np.mean(np.abs(jerk)))
    features["jerk_max_abs"] = float(np.max(np.abs(jerk)))

    # 4. Tilt angle (angle in degrees between the acceleration vector
    # and vertical) -- captures postural change (e.g. upright ->
    # horizontal during a fall) independent of magnitude.
    norm = np.linalg.norm(acc, axis=1)
    norm_safe = np.where(norm == 0, 1e-8, norm)  # guard divide-by-zero on a degenerate all-zero window
    cos_theta = np.clip(acc[:, 2] / norm_safe, -1.0, 1.0)
    tilt_deg = np.degrees(np.arccos(cos_theta))
    features["tilt_mean"] = float(np.mean(tilt_deg))
    features["tilt_std"] = float(np.std(tilt_deg))
    features["tilt_range"] = float(np.max(tilt_deg) - np.min(tilt_deg))

    # 5. Signal magnitude area (SMA) -- a standard HAR feature, mean of
    # summed absolute per-axis acceleration.
    features["sma"] = float(np.mean(np.sum(np.abs(acc), axis=1)))

    # 6. Frequency-domain features on the acceleration-magnitude signal
    # (3 features): a real fall's sharp, broadband transient looks very
    # different in frequency content from rhythmic walking/jogging.
    freqs, power = _power_spectrum(accel_mag, sample_rate_hz)
    if len(freqs) > 1:
        # Exclude the DC bin (index 0) -- already near-zero after mean
        # subtraction above, but excluded explicitly rather than relying
        # on floating-point cleanliness.
        ac_freqs, ac_power = freqs[1:], power[1:]
        total_power = float(np.sum(ac_power))
        if total_power > 0:
            dominant_freq = float(ac_freqs[np.argmax(ac_power)])
            probs = ac_power / total_power
            spectral_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        else:
            dominant_freq = 0.0
            spectral_entropy = 0.0
        spectral_energy = total_power
    else:
        dominant_freq = 0.0
        spectral_energy = 0.0
        spectral_entropy = 0.0
    features["dominant_freq_hz"] = dominant_freq
    features["spectral_energy"] = spectral_energy
    features["spectral_entropy"] = spectral_entropy

    return features


FEATURE_NAMES = list(compute_window_features(
    np.zeros((10, len(CHANNELS)), dtype=np.float32)
).keys())  # computed once, from a dummy window, so this list can never silently drift out of sync with compute_window_features itself


def compute_features_batch(
    windows_df: pd.DataFrame,
    windowing_config: Optional[WindowingConfig] = None,
    signal_cache: Optional[dict] = None,
) -> pd.DataFrame:
    """Compute features for every window in a windows manifest
    (`detection.dataset.build_windows_manifest`'s output).

    Returns a DataFrame with one row per window: all `FEATURE_NAMES`
    columns, plus `label`, `dataset`, `global_subject_id`,
    `activity_code`, `trial_id`, `window_index` carried through from
    the input for traceability and for `detection.split`'s
    subject-aware grouping.
    """
    windowing_config = windowing_config or WindowingConfig()
    signal_cache = signal_cache if signal_cache is not None else {}

    rows = []
    for _, window_row in windows_df.iterrows():
        window = load_window(
            window_row,
            window_length_samples=windowing_config.window_length_samples,
            signal_cache=signal_cache,
        )
        features = compute_window_features(window, sample_rate_hz=windowing_config.target_rate_hz)
        features.update({
            "label": window_row["label"],
            "dataset": window_row["dataset"],
            "global_subject_id": window_row["global_subject_id"],
            "activity_code": window_row["activity_code"],
            "trial_id": window_row["trial_id"],
            "window_index": window_row["window_index"],
        })
        rows.append(features)

    return pd.DataFrame(rows)
