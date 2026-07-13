"""Tests for prediction/features.py -- rolling auxiliary channels
(accel magnitude, jerk, tilt deviation)."""
import numpy as np
import pandas as pd
import pytest

from prediction.dataset import CHANNELS
from prediction.features import (
    AUX_CHANNEL_NAMES,
    augment_window,
    compute_auxiliary_channels,
    load_augmented_window,
)


def _window(acc, gyro=None, n=None):
    """Build a (n_samples, 6) window from an acc array (n,3); gyro
    defaults to zeros since these features don't use it."""
    acc = np.asarray(acc, dtype=np.float32)
    n = n or len(acc)
    gyro = np.zeros((n, 3), dtype=np.float32) if gyro is None else np.asarray(gyro, dtype=np.float32)
    return np.concatenate([acc, gyro], axis=1)


def test_output_shape_and_column_order():
    window = _window(np.ones((100, 3)))
    aux = compute_auxiliary_channels(window, sample_rate_hz=100.0)

    assert aux.shape == (100, 3)
    assert AUX_CHANNEL_NAMES == ["accel_mag", "jerk", "tilt_deviation_deg"]


def test_accel_magnitude_is_euclidean_norm():
    # A constant (3,4,0) vector -> magnitude 5 at every sample.
    acc = np.tile([3.0, 4.0, 0.0], (10, 1))
    window = _window(acc)

    aux = compute_auxiliary_channels(window)

    np.testing.assert_allclose(aux[:, 0], 5.0)


def test_purely_vertical_acceleration_has_zero_tilt_deviation():
    # Acceleration pointing entirely along canonical z -- by
    # construction (per the module docstring) this IS the calibrated
    # standing baseline, so deviation should be 0 degrees.
    acc = np.tile([0.0, 0.0, 1.0], (10, 1))
    window = _window(acc)

    aux = compute_auxiliary_channels(window)

    np.testing.assert_allclose(aux[:, 2], 0.0, atol=1e-4)


def test_purely_horizontal_acceleration_has_90_degree_tilt_deviation():
    acc = np.tile([1.0, 0.0, 0.0], (10, 1))
    window = _window(acc)

    aux = compute_auxiliary_channels(window)

    np.testing.assert_allclose(aux[:, 2], 90.0, atol=1e-4)


def test_jerk_is_zero_for_constant_magnitude_signal():
    acc = np.tile([0.0, 0.0, 1.0], (20, 1))
    window = _window(acc)

    aux = compute_auxiliary_channels(window)

    np.testing.assert_allclose(aux[:, 1], 0.0, atol=1e-4)


def test_jerk_matches_expected_slope_for_linear_ramp():
    # Magnitude ramps linearly from 0 to 1.98 over 100 samples at
    # 100Hz (acc_z goes 0.00, 0.02, 0.04, ...) -> constant slope of
    # 0.02 units/sample * 100 samples/s = 2.0 units/s everywhere
    # (central differences are exact for a linear ramp, including the
    # endpoints, which np.gradient handles via one-sided differences).
    n = 100
    acc_z = np.arange(n, dtype=np.float32) * 0.02
    acc = np.stack([np.zeros(n), np.zeros(n), acc_z], axis=1)
    window = _window(acc, n=n)

    aux = compute_auxiliary_channels(window, sample_rate_hz=100.0)

    np.testing.assert_allclose(aux[:, 1], 2.0, atol=1e-3)


def test_zero_acceleration_window_does_not_raise_and_yields_zero_norm():
    # Degenerate all-zero window -- divide-by-zero guarded (matches
    # detection/features.py's same guard), not expected on real data
    # but must not crash.
    acc = np.zeros((10, 3))
    window = _window(acc)

    aux = compute_auxiliary_channels(window)

    np.testing.assert_allclose(aux[:, 0], 0.0)
    assert np.all(np.isfinite(aux))


def test_too_few_samples_raises():
    window = _window(np.ones((1, 3)))
    with pytest.raises(ValueError, match="at least 2 samples"):
        compute_auxiliary_channels(window)


def test_wrong_channel_count_raises():
    bad_window = np.ones((10, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected 6 channels"):
        compute_auxiliary_channels(bad_window)


def test_augment_window_concatenates_raw_and_aux_channels():
    window = _window(np.tile([0.0, 0.0, 1.0], (50, 1)))

    augmented = augment_window(window)

    assert augmented.shape == (50, len(CHANNELS) + len(AUX_CHANNEL_NAMES))
    # First 6 columns are the untouched raw channels.
    np.testing.assert_allclose(augmented[:, :6], window)
    # Column 8 (tilt_deviation_deg) should be ~0 for this vertical window.
    np.testing.assert_allclose(augmented[:, 8], 0.0, atol=1e-4)


def _write_ramp_parquet(path, n_rows: int):
    df = pd.DataFrame({
        "time_s": np.arange(n_rows) / 100.0,
        "acc_x": np.zeros(n_rows, dtype=np.float32),
        "acc_y": np.zeros(n_rows, dtype=np.float32),
        "acc_z": np.ones(n_rows, dtype=np.float32),
        "gyro_x": np.zeros(n_rows, dtype=np.float32),
        "gyro_y": np.zeros(n_rows, dtype=np.float32),
        "gyro_z": np.zeros(n_rows, dtype=np.float32),
    })
    df.to_parquet(path, index=False)


def test_load_augmented_window_end_to_end(tmp_path):
    path = tmp_path / "trial.parquet"
    _write_ramp_parquet(path, n_rows=200)
    window_row = pd.Series({"harmonized_path": str(path), "start_frame": 0, "end_frame": 100})

    augmented = load_augmented_window(window_row, window_length_samples=100)

    assert augmented.shape == (100, 9)
    # acc_z constant at 1.0, acc_x/y at 0.0 -> accel_mag == 1.0 everywhere.
    np.testing.assert_allclose(augmented[:, 6], 1.0)
    # constant magnitude -> zero jerk.
    np.testing.assert_allclose(augmented[:, 7], 0.0, atol=1e-4)
    # purely vertical -> zero tilt deviation.
    np.testing.assert_allclose(augmented[:, 8], 0.0, atol=1e-4)
