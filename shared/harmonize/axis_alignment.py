"""Per-subject axis alignment (gravity-vector calibration).

Computes the rotation that maps a subject's raw, arbitrarily-tilted
sensor orientation onto a canonical vertical axis, using the mean
acceleration vector during a genuinely still segment (gravity is the
only thing acting on the sensor while stationary).

Calibration source priority (frozen design, see PROJECT_CHECKPOINT.md):
1. T01 ("stand still for 30s"), if present AND the Task 3.4 stationary
   detector confirms the subject was actually still for a large
   fraction of it -- don't trust the T01 label blindly.
2. Auto-detected stillness at the start of a standing-initiated trial
   (T02, T06-T09, T20-T21), if T01 is missing or fails that check.
3. Neither found -> returns None. Filling that gap with a group-average
   rotation is Task 3.6's job, not this module's.

This module is deliberately dataset-agnostic: it only requires objects
with a `.signal` DataFrame (with acc_x/y/z, gyro_x/y/z columns) and a
`.metadata.task_id` int -- it does not import KFall's reader directly,
so the same calibration logic will apply to SisFall/FallAllD later
without modification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from shared.harmonize.stationarity import detect_stationary_segment
from shared.harmonize.units import ACCEL_COLUMNS, GYRO_COLUMNS

# KFall task ID for the dedicated "stand still" trial.
T01_TASK_ID = 1

# Task IDs that begin from a standing posture before any motion starts
# -- candidates for the auto-detect fallback. (T20/T21 begin standing
# before stairs; the sit/lie-down tasks are deliberately excluded, since
# a seated or reclined torso has a different resting tilt than standing
# and would bias the calibration -- see PROJECT_CHECKPOINT.md.)
STANDING_INITIATED_TASK_IDS = frozenset({2, 6, 7, 8, 9, 20, 21})


@dataclass
class CalibrationResult:
    rotation: np.ndarray        # 3x3 rotation matrix
    source: str                 # "T01" | "auto_detected"
    gravity_vector: np.ndarray  # measured (pre-rotation) mean gravity direction


def compute_gravity_rotation(accel_segment: np.ndarray) -> np.ndarray:
    """Compute the rotation matrix that maps the mean direction of
    `accel_segment` (an (n, 3) array of accelerometer readings during a
    still period) onto the canonical vertical axis [0, 0, 1].

    Uses the standard vector-alignment (Rodrigues) formula. Handles the
    degenerate near-parallel and near-antiparallel cases explicitly.
    """
    mean_vec = accel_segment.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm < 1e-8:
        raise ValueError(
            "Mean acceleration magnitude is ~0 -- cannot determine a gravity "
            "direction from this segment (sensor may be free-falling, or the "
            "segment isn't actually a valid stationary window)."
        )

    a = mean_vec / norm
    b = np.array([0.0, 0.0, 1.0])

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)

    if s < 1e-8:
        if c > 0:
            # Already aligned with target.
            return np.eye(3)
        # Antiparallel: 180-degree rotation about any axis perpendicular to a.
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis = axis / np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)

    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    rotation = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return rotation


def apply_rotation(signal: pd.DataFrame, rotation: np.ndarray) -> pd.DataFrame:
    """Apply a rotation matrix to both the acceleration and (if present)
    gyroscope columns of `signal`. Returns a new DataFrame.

    Both quantities are vectors expressed in the sensor's frame, so the
    same frame-correcting rotation applies to each identically.
    """
    out = signal.copy()

    if all(c in signal.columns for c in ACCEL_COLUMNS):
        accel = signal[ACCEL_COLUMNS].to_numpy()
        out[ACCEL_COLUMNS] = accel @ rotation.T

    if all(c in signal.columns for c in GYRO_COLUMNS):
        gyro = signal[GYRO_COLUMNS].to_numpy()
        out[GYRO_COLUMNS] = gyro @ rotation.T

    return out


def _calibration_from_segment(signal: pd.DataFrame, start: int, end: int, source: str) -> CalibrationResult:
    accel_segment = signal[ACCEL_COLUMNS].to_numpy()[start:end]
    rotation = compute_gravity_rotation(accel_segment)
    return CalibrationResult(
        rotation=rotation,
        source=source,
        gravity_vector=accel_segment.mean(axis=0),
    )


def calibrate_subject(
    trials: list[Any],
    standing_initiated_task_ids: frozenset[int] = STANDING_INITIATED_TASK_IDS,
    sample_rate_hz: float = 100.0,
    min_duration_s: float = 2.0,
    t01_min_coverage_fraction: float = 0.5,
) -> Optional[CalibrationResult]:
    """Compute a subject's gravity-alignment calibration.

    `trials` is that subject's list of parsed trials (any object with
    `.signal` and `.metadata.task_id`). Tries T01 first, falls back to
    auto-detecting stillness in a standing-initiated trial. Returns
    None if neither succeeds -- see module docstring for what happens next.
    """
    t01_trial = next((t for t in trials if t.metadata.task_id == T01_TASK_ID), None)
    if t01_trial is not None:
        segment = detect_stationary_segment(t01_trial.signal, sample_rate_hz, min_duration_s)
        if segment is not None:
            start, end = segment
            coverage = (end - start) / len(t01_trial.signal)
            if coverage >= t01_min_coverage_fraction:
                return _calibration_from_segment(t01_trial.signal, start, end, source="T01")
            # T01 exists but the subject wasn't still for most of it --
            # don't trust it, fall through to auto-detection below.

    for trial in sorted(trials, key=lambda t: (t.metadata.task_id, t.metadata.trial_id)):
        if trial.metadata.task_id not in standing_initiated_task_ids:
            continue
        segment = detect_stationary_segment(trial.signal, sample_rate_hz, min_duration_s)
        if segment is not None:
            start, end = segment
            return _calibration_from_segment(trial.signal, start, end, source="auto_detected")

    return None
