"""Tests for shared/harmonize/orchestration.py (Stage 3, Task 3.10).

Runs the full pipeline end-to-end against the synthetic fixture set
(NOT real KFall data -- that's a separate manual run against your real
downloaded files once this passes). The fixture set was specifically
extended for this task to include:
  - SA06: has a genuine T01 (stand-still) trial -> should calibrate via T01
  - SA07: has NO T01, but has a standing-initiated T02 trial with a
    quiet start -> should calibrate via auto_detected

This means the test genuinely exercises both the primary and
auto-detect calibration tiers together in one run, not just in isolation.
"""
from pathlib import Path

import pandas as pd
import pytest

from shared.harmonize.orchestration import run_harmonization
from shared.harmonize.pipeline import HarmonizationConfig
from shared.manifest import load_manifest

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "kfall_mock"
SENSOR_ROOT = FIXTURE_ROOT / "sensor_data"
LABEL_ROOT = FIXTURE_ROOT / "label_data"


def test_unregistered_dataset_raises_clear_error(tmp_path):
    with pytest.raises(ValueError, match="No trial loader registered"):
        run_harmonization(
            dataset="sisfall",
            sensor_root=SENSOR_ROOT, label_root=LABEL_ROOT,
            harmonized_root=tmp_path / "harmonized", quarantine_root=tmp_path / "quarantine",
            harmonization_config=HarmonizationConfig(),
        )


def test_end_to_end_processes_all_fixture_trials(tmp_path):
    harmonized_root = tmp_path / "harmonized"
    quarantine_root = tmp_path / "quarantine"

    summary = run_harmonization(
        dataset="kfall",
        sensor_root=SENSOR_ROOT, label_root=LABEL_ROOT,
        harmonized_root=harmonized_root, quarantine_root=quarantine_root,
        harmonization_config=HarmonizationConfig(),
    )

    assert summary.n_trials_total == 5  # SA06: T01,T05,T22 ; SA07: T02,T05
    assert summary.n_written + summary.n_quarantined == 5


def test_end_to_end_exercises_t01_and_auto_detect_tiers(tmp_path):
    summary = run_harmonization(
        dataset="kfall",
        sensor_root=SENSOR_ROOT, label_root=LABEL_ROOT,
        harmonized_root=tmp_path / "harmonized", quarantine_root=tmp_path / "quarantine",
        harmonization_config=HarmonizationConfig(),
    )

    # SA06 calibrates via its real T01; SA07 has none, falls back to
    # auto-detect on T02. Both tiers should show up, group_fallback
    # should NOT be needed since every subject in this batch succeeds
    # on its own.
    assert summary.calibration_source_counts.get("T01") == 1
    assert summary.calibration_source_counts.get("auto_detected") == 1
    assert "group_fallback" not in summary.calibration_source_counts


def test_end_to_end_writes_harmonized_files_to_disk(tmp_path):
    harmonized_root = tmp_path / "harmonized"
    run_harmonization(
        dataset="kfall",
        sensor_root=SENSOR_ROOT, label_root=LABEL_ROOT,
        harmonized_root=harmonized_root, quarantine_root=tmp_path / "quarantine",
        harmonization_config=HarmonizationConfig(),
    )

    written_files = list((harmonized_root / "kfall").glob("*.parquet"))
    assert len(written_files) >= 1

    sample = pd.read_parquet(written_files[0])
    assert list(sample.columns) == ["time_s", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def test_end_to_end_writes_manifest(tmp_path):
    manifest_path = tmp_path / "harmonized" / "manifest.parquet"

    run_harmonization(
        dataset="kfall",
        sensor_root=SENSOR_ROOT, label_root=LABEL_ROOT,
        harmonized_root=tmp_path / "harmonized", quarantine_root=tmp_path / "quarantine",
        harmonization_config=HarmonizationConfig(),
        manifest_path=manifest_path,
    )

    manifest_df = load_manifest(manifest_path)
    assert len(manifest_df) == 5
    assert set(manifest_df["subject_id"]) == {"SA06", "SA07"}


def test_end_to_end_manifest_skipped_when_no_path_given(tmp_path):
    # manifest_path is optional -- confirms it doesn't error or write
    # anything when omitted.
    run_harmonization(
        dataset="kfall",
        sensor_root=SENSOR_ROOT, label_root=LABEL_ROOT,
        harmonized_root=tmp_path / "harmonized", quarantine_root=tmp_path / "quarantine",
        harmonization_config=HarmonizationConfig(),
    )
    assert not (tmp_path / "harmonized" / "manifest.parquet").exists()
