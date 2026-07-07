"""Tests for the group-average fallback in shared/harmonize/axis_alignment.py
(Stage 3, Task 3.6).
"""
import numpy as np
import pytest

from shared.harmonize.axis_alignment import (
    CalibrationResult,
    resolve_group_fallback,
    summarize_calibration_sources,
)


def _make_calibration(source: str, gravity_direction) -> CalibrationResult:
    gravity_vector = np.array(gravity_direction, dtype=float)
    gravity_vector = gravity_vector / np.linalg.norm(gravity_vector)
    # Rotation content doesn't matter for these tests -- identity is fine,
    # since resolve_group_fallback only reads .gravity_vector from valid
    # entries, not .rotation.
    return CalibrationResult(rotation=np.eye(3), source=source, gravity_vector=gravity_vector)


def test_missing_subjects_get_group_fallback_source():
    per_subject = {
        "SA06": _make_calibration("T01", [0.0, -1.0, 0.0]),
        "SA07": _make_calibration("auto_detected", [0.0, -1.0, 0.02]),
        "SA08": None,
    }

    resolved = resolve_group_fallback(per_subject)

    assert resolved["SA06"].source == "T01"
    assert resolved["SA07"].source == "auto_detected"
    assert resolved["SA08"].source == "group_fallback"


def test_already_valid_subjects_are_unchanged():
    cal_06 = _make_calibration("T01", [0.0, -1.0, 0.0])
    per_subject = {"SA06": cal_06, "SA07": None}

    resolved = resolve_group_fallback(per_subject)

    assert resolved["SA06"] is cal_06  # untouched, same object


def test_all_valid_returns_equivalent_dict():
    per_subject = {
        "SA06": _make_calibration("T01", [0.0, -1.0, 0.0]),
        "SA07": _make_calibration("T01", [0.0, -1.0, 0.0]),
    }

    resolved = resolve_group_fallback(per_subject)

    assert resolved["SA06"].source == "T01"
    assert resolved["SA07"].source == "T01"


def test_all_none_raises_clear_error():
    per_subject = {"SA06": None, "SA07": None}

    with pytest.raises(ValueError, match="no subjects"):
        resolve_group_fallback(per_subject)


def test_group_fallback_rotation_aligns_to_vertical():
    # Two subjects with slightly different but similar tilts calibrate
    # successfully; the fallback subject's rotation should still align
    # the AVERAGE of those directions to the vertical axis. Note: the
    # average of two non-identical unit vectors is itself shorter than
    # length 1 (basic vector geometry), so the correctness check here
    # is alignment (x/y components vanish), not magnitude == 1.
    per_subject = {
        "SA06": _make_calibration("T01", [0.1, -0.99, 0.0]),
        "SA07": _make_calibration("T01", [-0.1, -0.99, 0.0]),
        "SA08": None,
    }

    resolved = resolve_group_fallback(per_subject)
    fallback = resolved["SA08"]

    rotated = fallback.rotation @ fallback.gravity_vector
    assert rotated[0] == pytest.approx(0.0, abs=1e-6)
    assert rotated[1] == pytest.approx(0.0, abs=1e-6)
    assert rotated[2] > 0.99  # positive and close to the (sub-1) averaged magnitude


def test_summarize_calibration_sources_counts_correctly():
    resolved = {
        "SA06": _make_calibration("T01", [0, -1, 0]),
        "SA07": _make_calibration("T01", [0, -1, 0]),
        "SA08": _make_calibration("auto_detected", [0, -1, 0]),
        "SA09": _make_calibration("group_fallback", [0, -1, 0]),
    }

    counts = summarize_calibration_sources(resolved)

    assert counts == {"T01": 2, "auto_detected": 1, "group_fallback": 1}
