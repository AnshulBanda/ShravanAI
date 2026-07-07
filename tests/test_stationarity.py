"""Tests for shared/harmonize/stationarity.py (Stage 3, Task 3.4)."""
import numpy as np
import pandas as pd

from shared.harmonize.stationarity import detect_stationary_segment

SAMPLE_RATE_HZ = 100.0


def _make_signal(accel_segments, gyro_segments) -> pd.DataFrame:
    """Build a synthetic signal from a list of (n_samples, kind) pairs
    for accel and gyro, where kind is 'still' or 'moving'.
    """
    rng = np.random.default_rng(42)
    accel_rows, gyro_rows = [], []

    for n, kind in accel_segments:
        if kind == "still":
            accel_rows.append(rng.normal(0, 0.01, size=(n, 3)))
        else:
            t = np.arange(n) / SAMPLE_RATE_HZ
            wave = np.stack([np.sin(2 * np.pi * 3 * t + phase) for phase in (0, 1, 2)], axis=1)
            accel_rows.append(wave + rng.normal(0, 0.05, size=(n, 3)))

    for n, kind in gyro_segments:
        if kind == "still":
            gyro_rows.append(rng.normal(0, 0.5, size=(n, 3)))
        else:
            gyro_rows.append(rng.normal(0, 20.0, size=(n, 3)))

    accel = np.concatenate(accel_rows, axis=0)
    gyro = np.concatenate(gyro_rows, axis=0)
    n_total = len(accel)

    return pd.DataFrame({
        "time_s": np.arange(n_total) / SAMPLE_RATE_HZ,
        "acc_x": accel[:, 0], "acc_y": accel[:, 1], "acc_z": accel[:, 2],
        "gyro_x": gyro[:, 0], "gyro_y": gyro[:, 1], "gyro_z": gyro[:, 2],
    })


def test_finds_still_segment_embedded_in_movement():
    # 1s moving, 2s still, 2s moving -- still segment spans [100, 300)
    segments = [(100, "moving"), (200, "still"), (200, "moving")]
    signal = _make_signal(segments, segments)

    result = detect_stationary_segment(signal, sample_rate_hz=SAMPLE_RATE_HZ, min_duration_s=2.0)

    assert result is not None
    start, end = result
    assert abs(start - 100) <= 10
    assert abs(end - 300) <= 10
    assert end - start >= 200


def test_all_movement_returns_none():
    segments = [(500, "moving")]
    signal = _make_signal(segments, segments)

    result = detect_stationary_segment(signal, sample_rate_hz=SAMPLE_RATE_HZ, min_duration_s=2.0)
    assert result is None


def test_all_still_returns_nearly_full_range():
    segments = [(400, "still")]
    signal = _make_signal(segments, segments)

    result = detect_stationary_segment(signal, sample_rate_hz=SAMPLE_RATE_HZ, min_duration_s=2.0)

    assert result is not None
    start, end = result
    assert start <= 5
    assert end >= len(signal) - 5


def test_segment_shorter_than_min_duration_not_returned():
    # Only 1s of stillness (100 samples), surrounded by movement --
    # shorter than the 2.0s minimum, so no window should qualify.
    segments = [(150, "moving"), (100, "still"), (150, "moving")]
    signal = _make_signal(segments, segments)

    result = detect_stationary_segment(signal, sample_rate_hz=SAMPLE_RATE_HZ, min_duration_s=2.0)
    assert result is None


def test_signal_shorter_than_min_duration_returns_none():
    segments = [(50, "still")]  # only 0.5s, less than the 2.0s minimum
    signal = _make_signal(segments, segments)

    result = detect_stationary_segment(signal, sample_rate_hz=SAMPLE_RATE_HZ, min_duration_s=2.0)
    assert result is None


def test_longest_qualifying_segment_is_chosen_when_multiple_exist():
    # Two still segments of different lengths: [100,250) is 150 samples
    # (1.5s, below min_duration), [350, 600) is 250 samples (2.5s,
    # qualifies). Only the longer one should be returned.
    segments = [
        (100, "moving"),
        (150, "still"),   # too short to qualify alone
        (100, "moving"),
        (250, "still"),   # qualifies
        (100, "moving"),
    ]
    signal = _make_signal(segments, segments)

    result = detect_stationary_segment(signal, sample_rate_hz=SAMPLE_RATE_HZ, min_duration_s=2.0)

    assert result is not None
    start, end = result
    expected_start = 100 + 150 + 100  # 350
    expected_end = expected_start + 250  # 600
    assert abs(start - expected_start) <= 10
    assert abs(end - expected_end) <= 10
