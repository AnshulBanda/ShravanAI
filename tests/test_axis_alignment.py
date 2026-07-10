"""Tests for shared/harmonize/axis_alignment.py (Stage 3, Task 3.5)."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from shared.harmonize.axis_alignment import (
    CalibrationResult,
    apply_rotation,
    calibrate_subject,
    compute_gravity_rotation,
)

SAMPLE_RATE_HZ = 100.0


# Minimal trial-like fixtures -- deliberately NOT importing KFall's
# ParsedTrial/TrialMetadata, to keep this test honest about the fact
# that axis_alignment.py only depends on `.signal` and
# `.metadata.task_id`/`.metadata.trial_id`, not any KFall-specific type.
@dataclass
class _FakeMetadata:
    task_id: int
    trial_id: str = "R01"


@dataclass
class _FakeTrial:
    signal: pd.DataFrame
    metadata: _FakeMetadata


def _make_signal(accel_segments, gyro_segments=None) -> pd.DataFrame:
    """accel_segments: list of (n_samples, kind, tilt_vector) where kind
    is 'still' or 'moving'. tilt_vector is the direction gravity points
    in during 'still' segments (need not be unit-normalized).
    """
    rng = np.random.default_rng(7)
    accel_rows = []
    for n, kind, tilt in accel_segments:
        if kind == "still":
            direction = np.array(tilt, dtype=float)
            direction = direction / np.linalg.norm(direction)
            accel_rows.append(direction + rng.normal(0, 0.01, size=(n, 3)))
        else:
            t = np.arange(n) / SAMPLE_RATE_HZ
            wave = np.stack([np.sin(2 * np.pi * 3 * t + p) for p in (0, 1, 2)], axis=1)
            accel_rows.append(wave + rng.normal(0, 0.05, size=(n, 3)))
    accel = np.concatenate(accel_rows, axis=0)
    n_total = len(accel)

    if gyro_segments is None:
        gyro = rng.normal(0, 0.5, size=(n_total, 3))
    else:
        gyro_rows = []
        for n, kind in gyro_segments:
            if kind == "still":
                gyro_rows.append(rng.normal(0, 0.5, size=(n, 3)))
            else:
                gyro_rows.append(rng.normal(0, 20.0, size=(n, 3)))
        gyro = np.concatenate(gyro_rows, axis=0)

    return pd.DataFrame({
        "time_s": np.arange(n_total) / SAMPLE_RATE_HZ,
        "acc_x": accel[:, 0], "acc_y": accel[:, 1], "acc_z": accel[:, 2],
        "gyro_x": gyro[:, 0], "gyro_y": gyro[:, 1], "gyro_z": gyro[:, 2],
    })


# --- compute_gravity_rotation ---

def test_compute_gravity_rotation_aligns_tilted_gravity_to_vertical():
    # Gravity tilted ~30 degrees off pure-vertical, matching a real
    # arbitrary mounting angle scenario.
    tilt = np.array([0.5, 0.0, 0.866])  # unit vector, 30 deg off z-axis
    rng = np.random.default_rng(0)
    accel_segment = tilt + rng.normal(0, 0.005, size=(200, 3))

    rotation = compute_gravity_rotation(accel_segment)
    rotated_mean = rotation @ accel_segment.mean(axis=0)

    assert rotated_mean[2] == pytest.approx(1.0, abs=0.02)
    assert rotated_mean[0] == pytest.approx(0.0, abs=0.02)
    assert rotated_mean[1] == pytest.approx(0.0, abs=0.02)


def test_compute_gravity_rotation_handles_negative_axis_gravity():
    # Matches the real KFall SA06 finding: gravity on acc_y at ~-1.0,
    # not acc_z. Confirms the rotation correctly handles a
    # near-antiparallel-to-other-axis case.
    tilt = np.array([0.0, -1.0, 0.0])
    rng = np.random.default_rng(1)
    accel_segment = tilt + rng.normal(0, 0.005, size=(200, 3))

    rotation = compute_gravity_rotation(accel_segment)
    rotated_mean = rotation @ accel_segment.mean(axis=0)

    assert rotated_mean[2] == pytest.approx(1.0, abs=0.02)


def test_compute_gravity_rotation_handles_already_aligned_case():
    tilt = np.array([0.0, 0.0, 1.0])
    rng = np.random.default_rng(2)
    accel_segment = tilt + rng.normal(0, 0.005, size=(200, 3))

    rotation = compute_gravity_rotation(accel_segment)
    rotated_mean = rotation @ accel_segment.mean(axis=0)
    assert rotated_mean[2] == pytest.approx(1.0, abs=0.02)


def test_compute_gravity_rotation_handles_exact_antiparallel_case():
    # Gravity pointing exactly along -z -- the degenerate 180-degree
    # rotation branch.
    accel_segment = np.tile([0.0, 0.0, -1.0], (50, 1))

    rotation = compute_gravity_rotation(accel_segment)
    rotated_mean = rotation @ accel_segment.mean(axis=0)
    assert rotated_mean[2] == pytest.approx(1.0, abs=1e-6)


def test_compute_gravity_rotation_raises_on_degenerate_zero_vector():
    accel_segment = np.zeros((50, 3))
    with pytest.raises(ValueError):
        compute_gravity_rotation(accel_segment)


# --- apply_rotation ---

def test_apply_rotation_rotates_accel_and_gyro_consistently():
    signal = pd.DataFrame({
        "time_s": [0.0],
        "acc_x": [1.0], "acc_y": [0.0], "acc_z": [0.0],
        "gyro_x": [1.0], "gyro_y": [0.0], "gyro_z": [0.0],
    })
    # 90-degree rotation about z: x-axis -> y-axis
    rotation = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ])
    out = apply_rotation(signal, rotation)

    assert out["acc_x"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert out["acc_y"].iloc[0] == pytest.approx(1.0, abs=1e-9)
    assert out["gyro_x"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert out["gyro_y"].iloc[0] == pytest.approx(1.0, abs=1e-9)


def test_apply_rotation_does_not_mutate_input():
    signal = pd.DataFrame({
        "time_s": [0.0], "acc_x": [1.0], "acc_y": [0.0], "acc_z": [0.0],
        "gyro_x": [0.0], "gyro_y": [0.0], "gyro_z": [0.0],
    })
    original = signal.copy()
    _ = apply_rotation(signal, np.eye(3))
    pd.testing.assert_frame_equal(signal, original)


# --- calibrate_subject ---

def test_calibrate_subject_uses_t01_when_clean():
    t01_signal = _make_signal([(300, "still", [0.0, 0.0, 1.0])])
    trials = [_FakeTrial(signal=t01_signal, metadata=_FakeMetadata(task_id=1))]

    result = calibrate_subject(trials)

    assert result is not None
    assert result.source == "T01"


def test_calibrate_subject_falls_back_when_t01_missing():
    # No T01 at all; task 2 (standing-initiated) has a clean quiet start.
    t02_signal = _make_signal([
        (250, "still", [0.0, -1.0, 0.0]),
        (150, "moving", None),
    ])
    trials = [_FakeTrial(signal=t02_signal, metadata=_FakeMetadata(task_id=2))]

    result = calibrate_subject(trials)

    assert result is not None
    assert result.source == "auto_detected"


def test_calibrate_subject_falls_back_when_t01_mostly_fidgety():
    # T01 exists but the subject was still for only a small fraction of
    # it -- should NOT be trusted, should fall through to task 6 instead.
    t01_signal = _make_signal([
        (250, "moving", None),
        (50, "still", [0.0, 0.0, 1.0]),   # only 0.5s still, out of 3s total
    ])
    t06_signal = _make_signal([
        (250, "still", [0.0, 0.0, 1.0]),
        (150, "moving", None),
    ])
    trials = [
        _FakeTrial(signal=t01_signal, metadata=_FakeMetadata(task_id=1)),
        _FakeTrial(signal=t06_signal, metadata=_FakeMetadata(task_id=6)),
    ]

    result = calibrate_subject(trials, min_duration_s=0.4)

    assert result is not None
    assert result.source == "auto_detected"


def test_calibrate_subject_returns_none_when_nothing_usable():
    # Only moving trials, no T01, nothing standing-initiated with a
    # usable quiet segment.
    moving_signal = _make_signal([(300, "moving", None)])
    trials = [_FakeTrial(signal=moving_signal, metadata=_FakeMetadata(task_id=2))]

    result = calibrate_subject(trials)
    assert result is None


def test_calibrate_subject_ignores_non_standing_initiated_tasks_in_fallback():
    # Task 11 ("sit upright for 30s") is deliberately excluded from the
    # standing-initiated fallback set, even though it's a perfectly
    # still trial -- sitting posture has a different resting tilt than
    # standing and shouldn't be used as a standing calibration proxy.
    t11_signal = _make_signal([(300, "still", [0.3, 0.0, 0.95])])
    trials = [_FakeTrial(signal=t11_signal, metadata=_FakeMetadata(task_id=11))]

    result = calibrate_subject(trials)
    assert result is None


# --- primary_calibration_task_id (Stage 5: SisFall has no T01-equivalent) ---

def test_calibrate_subject_primary_task_id_none_skips_task_id_1_entirely():
    # A trial that would look exactly like a valid T01 (task_id=1,
    # genuinely still for its whole duration) must be IGNORED as a
    # primary-calibration candidate when primary_calibration_task_id is
    # explicitly None -- e.g. SisFall's D01 ("walking slowly") happens
    # to also be task_id 1, and must never be mistaken for a dedicated
    # calibration trial just because of that numeric coincidence.
    task_id_1_signal = _make_signal([(300, "still", [0.0, 0.0, 1.0])])
    standing_initiated_signal = _make_signal([
        (250, "still", [0.2, 0.0, 0.98]),
        (150, "moving", None),
    ])
    trials = [
        _FakeTrial(signal=task_id_1_signal, metadata=_FakeMetadata(task_id=1)),
        _FakeTrial(signal=standing_initiated_signal, metadata=_FakeMetadata(task_id=7)),
    ]

    result = calibrate_subject(
        trials,
        standing_initiated_task_ids=frozenset({7}),
        primary_calibration_task_id=None,
    )

    assert result is not None
    assert result.source == "auto_detected"
    # Sanity check it actually used task_id=7's gravity direction
    # ([0.2, 0, 0.98]-ish), not task_id=1's ([0, 0, 1]) -- confirms the
    # task_id=1 trial was genuinely skipped, not just deprioritized.
    np.testing.assert_allclose(
        result.gravity_vector / np.linalg.norm(result.gravity_vector),
        [0.2, 0.0, 0.98], atol=0.05,
    )


def test_calibrate_subject_primary_task_id_none_with_no_fallback_returns_none():
    # If the ONLY still trial available is task_id=1 and primary
    # calibration is disabled, and task_id=1 isn't in the
    # standing-initiated fallback set either, this must return None --
    # not silently fall back to using it anyway.
    task_id_1_signal = _make_signal([(300, "still", [0.0, 0.0, 1.0])])
    trials = [_FakeTrial(signal=task_id_1_signal, metadata=_FakeMetadata(task_id=1))]

    result = calibrate_subject(
        trials,
        standing_initiated_task_ids=frozenset({7, 8, 9}),  # deliberately excludes 1
        primary_calibration_task_id=None,
    )

    assert result is None


def test_calibrate_subject_default_primary_task_id_preserves_kfall_behavior():
    # Explicitly confirms the default argument still behaves exactly
    # like the pre-Stage-5 hardcoded version: task_id=1, sufficiently
    # still, with primary_calibration_task_id left at its default.
    t01_signal = _make_signal([(300, "still", [0.0, 0.0, 1.0])])
    trials = [_FakeTrial(signal=t01_signal, metadata=_FakeMetadata(task_id=1))]

    result = calibrate_subject(trials)

    assert result is not None
    assert result.source == "T01"
