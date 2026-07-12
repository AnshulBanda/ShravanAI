"""Tests for detection/features.py."""
import numpy as np
import pandas as pd
import pytest

from detection.dataset import CHANNELS
from detection.features import (
    FEATURE_NAMES,
    compute_features_batch,
    compute_window_features,
)
from detection.windowing import WindowingConfig


def _constant_window(values: list[float], n_rows: int = 200) -> np.ndarray:
    """A window where every row is identical: `values` (length 6)."""
    assert len(values) == len(CHANNELS)
    return np.tile(np.array(values, dtype=np.float32), (n_rows, 1))


def test_feature_names_matches_compute_window_features_keys():
    # FEATURE_NAMES is derived FROM compute_window_features itself, so
    # this is really just confirming that derivation didn't silently
    # break -- guards against someone hardcoding a stale copy later.
    window = _constant_window([0.1, 0.2, 0.3, 1.0, 2.0, 3.0])
    features = compute_window_features(window)

    assert list(features.keys()) == FEATURE_NAMES
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))  # no duplicate names


def test_constant_window_has_zero_std_and_zero_jerk():
    # A perfectly still window: no variation, no jerk.
    window = _constant_window([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # resting on acc_z (gravity)
    features = compute_window_features(window)

    assert features["acc_x_std"] == 0.0
    assert features["acc_z_std"] == 0.0
    assert features["jerk_mean_abs"] == 0.0
    assert features["jerk_max_abs"] == 0.0
    assert features["accel_mag_std"] == 0.0


def test_constant_window_mean_equals_the_constant_value():
    window = _constant_window([0.5, -0.5, 1.0, 10.0, -10.0, 5.0])
    features = compute_window_features(window)

    assert features["acc_x_mean"] == pytest.approx(0.5)
    assert features["acc_y_mean"] == pytest.approx(-0.5)
    assert features["gyro_x_mean"] == pytest.approx(10.0)


def test_gravity_only_window_has_zero_tilt():
    # Pure +1g on acc_z, nothing else: acceleration vector points
    # straight along vertical -- tilt angle should be ~0 degrees.
    window = _constant_window([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    features = compute_window_features(window)

    assert features["tilt_mean"] == pytest.approx(0.0, abs=1e-6)


def test_horizontal_gravity_window_has_90_degree_tilt():
    # +1g entirely on acc_x (device lying on its side): tilt from
    # vertical should be ~90 degrees.
    window = _constant_window([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    features = compute_window_features(window)

    assert features["tilt_mean"] == pytest.approx(90.0, abs=1e-4)


def test_sharp_spike_produces_large_jerk_and_range():
    # A quiet window with one sharp spike in the middle -- simulates a
    # fall impact within an otherwise-still window.
    window = _constant_window([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], n_rows=200)
    window = window.copy()
    window[100, 0] = 5.0  # single-sample spike on acc_x

    features = compute_window_features(window)

    assert features["jerk_max_abs"] > features["jerk_mean_abs"]
    assert features["acc_x_range"] == pytest.approx(5.0)
    assert features["accel_mag_max"] > features["accel_mag_mean"]


def test_zero_window_does_not_crash_on_degenerate_vectors():
    # All-zero window -- degenerate for tilt (0/0) and FFT (all-zero
    # signal). Must not raise, must not produce NaN/inf.
    window = np.zeros((200, len(CHANNELS)), dtype=np.float32)
    features = compute_window_features(window)

    for name, value in features.items():
        assert np.isfinite(value), f"{name} is not finite: {value}"


def test_oscillating_signal_has_nonzero_dominant_frequency():
    # A clean 5Hz sinusoid on acc_x (acc_z held at a constant 1g so the
    # window isn't oscillating around zero, like real resting gravity).
    #
    # The dominant-frequency feature is computed on accel_MAGNITUDE,
    # not the raw axis -- and sqrt(sin(2*pi*f*t)^2 + const^2) has its
    # fundamental period at 2f, not f (a standard property: squaring a
    # sinusoid doubles its apparent frequency, and sqrt doesn't undo
    # that). So a 5Hz input axis correctly shows up as a ~10Hz peak in
    # the magnitude's spectrum -- verified numerically before writing
    # this assertion, not assumed.
    t = np.arange(200) / 100.0
    window = np.zeros((200, len(CHANNELS)), dtype=np.float32)
    window[:, 0] = np.sin(2 * np.pi * 5.0 * t)
    window[:, 2] = 1.0  # resting gravity on z so accel isn't purely oscillating around 0

    features = compute_window_features(window, sample_rate_hz=100.0)

    assert features["dominant_freq_hz"] == pytest.approx(10.0, abs=0.5)
    assert features["spectral_energy"] > 0


# --- compute_features_batch ---

def test_compute_features_batch_produces_expected_columns(tmp_path):
    n_rows = 200
    df = pd.DataFrame({
        "time_s": np.arange(n_rows) / 100.0,
        **{col: np.random.default_rng(0).normal(0, 1, n_rows).astype(np.float32) for col in CHANNELS},
    })
    path = tmp_path / "trial.parquet"
    df.to_parquet(path, index=False)

    windows_df = pd.DataFrame([{
        "dataset": "kfall", "global_subject_id": "kfall_SA06", "activity_code": "T01",
        "trial_id": "R01", "label": 0, "window_index": 0,
        "start_frame": 0, "end_frame": 200, "harmonized_path": str(path),
    }])

    features_df = compute_features_batch(windows_df, windowing_config=WindowingConfig())

    assert len(features_df) == 1
    for name in FEATURE_NAMES:
        assert name in features_df.columns
    assert features_df["label"].iloc[0] == 0
    assert features_df["global_subject_id"].iloc[0] == "kfall_SA06"


def test_compute_features_batch_reuses_cache_across_windows(tmp_path):
    n_rows = 300
    df = pd.DataFrame({
        "time_s": np.arange(n_rows) / 100.0,
        **{col: np.arange(n_rows, dtype=np.float32) for col in CHANNELS},
    })
    path = tmp_path / "trial.parquet"
    df.to_parquet(path, index=False)

    windows_df = pd.DataFrame([
        {"dataset": "kfall", "global_subject_id": "kfall_SA06", "activity_code": "T01",
         "trial_id": "R01", "label": 0, "window_index": 0,
         "start_frame": 0, "end_frame": 200, "harmonized_path": str(path)},
        {"dataset": "kfall", "global_subject_id": "kfall_SA06", "activity_code": "T01",
         "trial_id": "R01", "label": 0, "window_index": 1,
         "start_frame": 100, "end_frame": 300, "harmonized_path": str(path)},
    ])

    cache: dict = {}
    features_df = compute_features_batch(windows_df, windowing_config=WindowingConfig(), signal_cache=cache)

    assert len(features_df) == 2
    assert len(cache) == 1  # same file, read once
