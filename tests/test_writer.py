"""Tests for shared/harmonize/writer.py (Stage 3, Task 3.9)."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from shared.harmonize.writer import write_harmonized_trial


@dataclass
class _FakeMetadata:
    dataset: str = "kfall"
    subject_id: str = "SA06"
    activity_code: str = "T22"
    trial_id: str = "R01"
    label: str = "fall"
    fall_onset_frame: Optional[int] = 130
    fall_impact_frame: Optional[int] = 208


def _sample_signal(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    t = np.arange(n) / 100.0
    return pd.DataFrame({
        "time_s": t,
        "acc_x": rng.normal(0, 0.1, n), "acc_y": rng.normal(0, 0.1, n), "acc_z": rng.normal(0, 0.1, n),
        "gyro_x": rng.normal(0, 1, n), "gyro_y": rng.normal(0, 1, n), "gyro_z": rng.normal(0, 1, n),
    })


def test_valid_trial_written_to_harmonized_root(tmp_path):
    harmonized_root = tmp_path / "harmonized"
    quarantine_root = tmp_path / "quarantine"
    signal = _sample_signal()

    out_path = write_harmonized_trial(
        signal, _FakeMetadata(), calibration_source="T01", issues=[],
        harmonized_root=harmonized_root, quarantine_root=quarantine_root,
    )

    assert out_path == harmonized_root / "kfall" / "SA06_T22_R01.parquet"
    assert out_path.exists()
    assert not (quarantine_root / "kfall").exists()


def test_quarantined_trial_written_to_quarantine_root(tmp_path):
    harmonized_root = tmp_path / "harmonized"
    quarantine_root = tmp_path / "quarantine"
    signal = _sample_signal()

    out_path = write_harmonized_trial(
        signal, _FakeMetadata(), calibration_source="T01",
        issues=["schema mismatch"],
        harmonized_root=harmonized_root, quarantine_root=quarantine_root,
    )

    assert out_path == quarantine_root / "kfall" / "SA06_T22_R01.parquet"
    assert out_path.exists()
    assert not (harmonized_root / "kfall").exists()


def test_round_trip_values_match(tmp_path):
    signal = _sample_signal()
    out_path = write_harmonized_trial(
        signal, _FakeMetadata(), calibration_source="T01", issues=[],
        harmonized_root=tmp_path / "harmonized", quarantine_root=tmp_path / "quarantine",
    )

    read_back = pd.read_parquet(out_path)
    pd.testing.assert_frame_equal(read_back, signal, check_exact=False, atol=1e-9)


def test_sidecar_json_contains_expected_provenance(tmp_path):
    signal = _sample_signal()
    out_path = write_harmonized_trial(
        signal, _FakeMetadata(), calibration_source="auto_detected", issues=[],
        harmonized_root=tmp_path / "harmonized", quarantine_root=tmp_path / "quarantine",
        provenance_extra={"target_rate_hz": 100.0, "filter_low_hz": 0.5, "filter_high_hz": 20.0, "filter_order": 4},
    )

    provenance = json.loads(out_path.with_suffix(".json").read_text())
    assert provenance["dataset"] == "kfall"
    assert provenance["subject_id"] == "SA06"
    assert provenance["calibration_source"] == "auto_detected"
    assert provenance["fall_onset_frame"] == 130
    assert provenance["fall_impact_frame"] == 208
    assert provenance["accepted"] is True
    assert provenance["issues"] == []
    assert provenance["target_rate_hz"] == 100.0
    assert provenance["filter_low_hz"] == 0.5


def test_sidecar_json_logs_issues_for_quarantined_trial(tmp_path):
    signal = _sample_signal()
    issues = ["Schema mismatch: missing gyro_z", "NaN values found in column(s): ['acc_x']"]
    out_path = write_harmonized_trial(
        signal, _FakeMetadata(), calibration_source="T01", issues=issues,
        harmonized_root=tmp_path / "harmonized", quarantine_root=tmp_path / "quarantine",
    )

    provenance = json.loads(out_path.with_suffix(".json").read_text())
    assert provenance["accepted"] is False
    assert provenance["issues"] == issues


def test_different_subjects_and_trials_do_not_collide(tmp_path):
    harmonized_root = tmp_path / "harmonized"
    quarantine_root = tmp_path / "quarantine"

    path1 = write_harmonized_trial(
        _sample_signal(), _FakeMetadata(subject_id="SA06", trial_id="R01"),
        calibration_source="T01", issues=[],
        harmonized_root=harmonized_root, quarantine_root=quarantine_root,
    )
    path2 = write_harmonized_trial(
        _sample_signal(), _FakeMetadata(subject_id="SA06", trial_id="R02"),
        calibration_source="T01", issues=[],
        harmonized_root=harmonized_root, quarantine_root=quarantine_root,
    )

    assert path1 != path2
    assert path1.exists() and path2.exists()
