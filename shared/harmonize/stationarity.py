"""Stationary-segment detection.

Generic function to find a quiet/still window within any harmonized
signal, based on rolling acceleration variance and gyroscope magnitude
staying below thresholds for a minimum duration.

Used two ways in Stage 3:
1. Validating that a subject's T01 ("stand still") trial is actually
   still, rather than trusting the label blindly (Task 3.5).
2. Falling back to auto-detecting a quiet window at the start of a
   standing-initiated trial (T02, T06-T09, T20-T21) when T01 is
   missing or fails that validation (Task 3.5/3.6).

Deliberately dataset- and task-agnostic: it only looks at the signal
itself, so it works identically regardless of which trial or dataset
it's handed.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

ACCEL_COLUMNS = ["acc_x", "acc_y", "acc_z"]
GYRO_COLUMNS = ["gyro_x", "gyro_y", "gyro_z"]


def detect_stationary_segment(
    signal: pd.DataFrame,
    sample_rate_hz: float,
    min_duration_s: float = 2.0,
    accel_var_threshold: float = 0.01,
    gyro_mag_threshold: float = 5.0,
) -> Optional[tuple[int, int]]:
    """Find the longest window where acceleration variance and gyro
    magnitude both stay below their thresholds for at least
    `min_duration_s` seconds.

    Returns (start_idx, end_idx) of the longest qualifying window (as
    positional indices into `signal`), or None if no window of at
    least `min_duration_s` qualifies anywhere in the signal.

    The variance/magnitude check is computed on a rolling basis using a
    window of `min_duration_s` seconds, then any run of consecutive
    rolling-windows that all pass the threshold is treated as one
    contiguous stationary segment (rather than requiring the whole
    signal to be still at once) -- this lets it find a short quiet
    period at the START of a longer, mostly-moving trial, which is
    exactly the auto-detect fallback's use case.
    """
    min_duration_samples = int(round(min_duration_s * sample_rate_hz))
    n = len(signal)
    if n < min_duration_samples:
        return None

    accel = signal[ACCEL_COLUMNS].to_numpy()
    gyro = signal[GYRO_COLUMNS].to_numpy()
    gyro_mag = np.linalg.norm(gyro, axis=1)

    # Rolling variance of acceleration magnitude (per-axis variance,
    # summed) and rolling mean gyro magnitude, both over a window of
    # min_duration_samples, evaluated at every possible start position.
    is_quiet = np.zeros(n, dtype=bool)
    for start in range(0, n - min_duration_samples + 1):
        end = start + min_duration_samples
        accel_var = accel[start:end].var(axis=0).sum()
        gyro_mean_mag = gyro_mag[start:end].mean()
        if accel_var < accel_var_threshold and gyro_mean_mag < gyro_mag_threshold:
            is_quiet[start:end] = True

    if not is_quiet.any():
        return None

    # Find the longest contiguous True run in is_quiet.
    best_start, best_len = None, 0
    run_start = None
    for i, quiet in enumerate(is_quiet):
        if quiet and run_start is None:
            run_start = i
        elif not quiet and run_start is not None:
            run_len = i - run_start
            if run_len > best_len:
                best_start, best_len = run_start, run_len
            run_start = None
    if run_start is not None:
        run_len = n - run_start
        if run_len > best_len:
            best_start, best_len = run_start, run_len

    if best_start is None or best_len < min_duration_samples:
        return None

    return best_start, best_start + best_len
