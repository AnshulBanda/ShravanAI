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
`.metadata.task_id` int -- it does not import KFall's reader directly.

Stage 5 update: this was ALMOST true from the start but not quite --
`calibrate_subject` used to hardcode `task_id == 1` as "the" primary
calibration trial with no way to opt out. That's fine for KFall (T01
really is task_id 1), but SisFall's D01 ("walking slowly") also happens
to be task_id 1 purely by coincidence of activity-code ordering, and
isn't a calibration trial at all -- SisFall has no dedicated
stand-still trial. `calibrate_subject` now takes an explicit
`primary_calibration_task_id` parameter (default `T01_TASK_ID`,
preserving KFall's exact prior behavior) so callers for datasets
without a primary calibration trial can pass `None` and skip straight
to auto-detection, instead of that skip happening as an accident of
numbering.
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
    primary_calibration_task_id: Optional[int] = T01_TASK_ID,
) -> Optional[CalibrationResult]:
    """Compute a subject's gravity-alignment calibration.

    `trials` is that subject's list of parsed trials (any object with
    `.signal` and `.metadata.task_id`). Tries the primary calibration
    trial first (KFall: T01, `task_id == 1` by default), falls back to
    auto-detecting stillness in a standing-initiated trial. Returns
    None if neither succeeds -- see module docstring for what happens next.

    `primary_calibration_task_id`: the task_id of this dataset's
    dedicated "stand still" trial, if it has one. Defaults to
    `T01_TASK_ID` (1) to match KFall's existing behavior exactly.
    **Pass `None` for any dataset that has no dedicated calibration
    trial** (e.g. SisFall, where every activity code is a real movement
    task and none is a reserved stand-still trial) -- this skips
    straight to the auto-detect fallback below, rather than risking a
    coincidental `task_id == 1` match on some unrelated activity being
    mislabeled with the KFall-specific "T01" calibration-source string.
    This parameter exists specifically because task_id numbering is NOT
    comparable across datasets (SisFall's D01 "walking slowly" happens
    to also have task_id 1, purely by coincidence of activity-code
    ordering) -- discovered while wiring SisFall into Stage 5, see
    PROJECT_CHECKPOINT.md.
    """
    if primary_calibration_task_id is not None:
        t01_trial = next(
            (t for t in trials if t.metadata.task_id == primary_calibration_task_id),
            None,
        )
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


def resolve_group_fallback(
    per_subject: dict[str, Optional[CalibrationResult]],
) -> dict[str, CalibrationResult]:
    """Fill in calibration for any subject where `calibrate_subject`
    returned None, using the average gravity direction across subjects
    who DID calibrate successfully (T01 or auto-detected).

    Assumes the sensor is mounted the same way across subjects in a
    given study protocol, so the average of everyone else's measured
    gravity direction is a reasonable stand-in for a subject with no
    usable stationary segment of their own -- a coarse approximation,
    which is exactly why this is the last-resort tier, not a first
    choice. Subjects already calibrated are returned unchanged.

    Raises ValueError if every subject is None (nothing to average from).
    """
    valid = {sid: cal for sid, cal in per_subject.items() if cal is not None}
    missing = {sid: cal for sid, cal in per_subject.items() if cal is None}

    if not missing:
        return dict(per_subject)

    if not valid:
        raise ValueError(
            "Cannot resolve group-average fallback: no subjects in this "
            "batch have a valid per-subject calibration to average from."
        )

    directions = np.array([
        cal.gravity_vector / np.linalg.norm(cal.gravity_vector)
        for cal in valid.values()
    ])
    mean_direction = directions.mean(axis=0)

    group_rotation = compute_gravity_rotation(mean_direction.reshape(1, -1))

    resolved = dict(valid)
    for subject_id in missing:
        resolved[subject_id] = CalibrationResult(
            rotation=group_rotation,
            source="group_fallback",
            gravity_vector=mean_direction,
        )
    return resolved


def summarize_calibration_sources(resolved: dict[str, CalibrationResult]) -> dict[str, int]:
    """One-line-summary helper: counts of each calibration source across
    a resolved (post-group-fallback) set of subjects. Intended for the
    human sanity check called for in the Stage 3 sprint plan -- if
    `group_fallback` is common rather than rare, something upstream
    (T01 parsing, or the stationarity thresholds) likely needs attention
    before trusting the harmonized output.
    """
    counts: dict[str, int] = {}
    for cal in resolved.values():
        counts[cal.source] = counts.get(cal.source, 0) + 1
    return counts
