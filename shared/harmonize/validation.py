"""Automated validation for harmonized trials.

Runs a set of sanity checks on a harmonized trial before it's accepted
(Task 3.9 wires the actual accept/quarantine routing based on this
function's output). Returns a list of human-readable issue strings --
empty list means the trial passed everything.

Design note on the calibration-sanity check specifically: it validates
`calibration.rotation @ calibration.gravity_vector` directly, NOT
anything about the final filtered signal. This is deliberate -- Task
3.7 established that the 0.5-20 Hz band-pass filter removes gravity's
near-DC content from the final harmonized output, so a persistent ~1g
bias is never expected to be present post-filter, even for a perfectly
calibrated trial. Checking the calibration object's own rotation-applied-
to-its-own-recorded-gravity-vector is the correct, filter-independent
way to confirm the alignment step itself was valid.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from shared.harmonize.axis_alignment import CalibrationResult
from shared.harmonize.units import ACCEL_COLUMNS, GYRO_COLUMNS

_EXPECTED_COLUMNS = ["time_s"] + ACCEL_COLUMNS + GYRO_COLUMNS

# Generous physical-plausibility bound. Real hardware (KFall's LPMS-B2,
# SisFall's ADXL345) tops out at +/-16g raw range; filtered output
# should always be well inside that, so anything beyond 20g strongly
# suggests corruption rather than a genuine (if extreme) fall impact.
_MAX_PLAUSIBLE_ACCEL_G = 20.0

# A column with essentially zero variance across an entire real trial
# (not just a short calibration window) almost certainly indicates
# sensor dropout, not genuine stillness -- real hardware noise floor
# should always produce SOME variance.
_FLATLINE_STD_THRESHOLD = 1e-6


def validate_harmonized_trial(
    signal: pd.DataFrame,
    metadata: Any,
    calibration: CalibrationResult,
    expected_rate_hz: float,
    expected_duration_range_s: Optional[tuple[float, float]] = None,
) -> list[str]:
    """Run all validation checks on one harmonized trial.

    `metadata` is duck-typed: checks that need `label`,
    `fall_onset_frame`, `fall_impact_frame` only run if those attributes
    are present, so this works across datasets that may not have all of
    them (e.g. SisFall/FallAllD have no frame-level labels at all).
    """
    issues: list[str] = []

    issues += _check_schema(signal)
    issues += _check_no_nan_inf(signal)
    issues += _check_timing_integrity(signal, expected_rate_hz)
    issues += _check_physical_plausibility(signal)
    issues += _check_calibration_sanity(calibration)
    issues += _check_duration_vs_protocol(signal, expected_rate_hz, expected_duration_range_s)
    issues += _check_label_consistency(signal, metadata)

    return issues


def _check_schema(signal: pd.DataFrame) -> list[str]:
    if list(signal.columns) != _EXPECTED_COLUMNS:
        return [
            f"Schema mismatch: expected columns {_EXPECTED_COLUMNS}, "
            f"got {list(signal.columns)}"
        ]
    return []


def _check_no_nan_inf(signal: pd.DataFrame) -> list[str]:
    issues = []
    numeric = signal.select_dtypes(include=[np.number])
    if numeric.isna().any().any():
        bad_cols = numeric.columns[numeric.isna().any()].tolist()
        issues.append(f"NaN values found in column(s): {bad_cols}")
    if np.isinf(numeric.to_numpy()).any():
        issues.append("Infinite values found in signal")
    return issues


def _check_timing_integrity(signal: pd.DataFrame, expected_rate_hz: float) -> list[str]:
    if "time_s" not in signal.columns or len(signal) < 2:
        return []
    diffs = np.diff(signal["time_s"].to_numpy())
    expected_step = 1.0 / expected_rate_hz
    if not np.allclose(diffs, expected_step, atol=expected_step * 0.1):
        return [
            f"Uneven or incorrect timestamp spacing: expected ~{expected_step:.5f}s "
            f"steps, found range [{diffs.min():.5f}, {diffs.max():.5f}]"
        ]
    return []


def _check_physical_plausibility(signal: pd.DataFrame) -> list[str]:
    issues = []
    for col in ACCEL_COLUMNS:
        if col not in signal.columns:
            continue
        values = signal[col].to_numpy()
        if np.abs(values).max() > _MAX_PLAUSIBLE_ACCEL_G:
            issues.append(
                f"{col}: implausible acceleration magnitude "
                f"{np.abs(values).max():.1f}g exceeds {_MAX_PLAUSIBLE_ACCEL_G}g bound"
            )
        if values.std() < _FLATLINE_STD_THRESHOLD:
            issues.append(f"{col}: flatline detected (std={values.std():.2e}), likely sensor dropout")
    return issues


def _check_calibration_sanity(calibration: CalibrationResult) -> list[str]:
    rotated = calibration.rotation @ calibration.gravity_vector
    issues = []
    if not (0.8 <= rotated[2] <= 1.2):
        issues.append(
            f"Calibration sanity failed: rotated gravity vertical component "
            f"{rotated[2]:.3f} outside expected [0.8, 1.2] range"
        )
    if abs(rotated[0]) > 0.2 or abs(rotated[1]) > 0.2:
        issues.append(
            f"Calibration sanity failed: rotated gravity horizontal components "
            f"({rotated[0]:.3f}, {rotated[1]:.3f}) not close enough to zero"
        )
    return issues


def _check_duration_vs_protocol(
    signal: pd.DataFrame,
    expected_rate_hz: float,
    expected_duration_range_s: Optional[tuple[float, float]],
) -> list[str]:
    if expected_duration_range_s is None:
        return []
    duration = len(signal) / expected_rate_hz
    low, high = expected_duration_range_s
    if not (low <= duration <= high):
        return [
            f"Trial duration {duration:.2f}s outside expected range "
            f"[{low}, {high}]s for this activity"
        ]
    return []


def _check_label_consistency(signal: pd.DataFrame, metadata: Any) -> list[str]:
    onset = getattr(metadata, "fall_onset_frame", None)
    impact = getattr(metadata, "fall_impact_frame", None)
    label = getattr(metadata, "label", None)

    issues = []
    if label == "adl":
        if onset is not None or impact is not None:
            issues.append(
                f"ADL trial has non-null onset/impact frames (onset={onset}, impact={impact})"
            )
        return issues

    if onset is None and impact is None:
        return issues  # nothing to check (e.g. label lookup found no match)

    if onset is None or impact is None:
        issues.append(f"Fall trial has only one of onset/impact set (onset={onset}, impact={impact})")
        return issues

    if not (onset < impact):
        issues.append(f"onset_frame ({onset}) is not before impact_frame ({impact})")
    if impact > len(signal):
        issues.append(f"impact_frame ({impact}) exceeds signal length ({len(signal)})")

    return issues
