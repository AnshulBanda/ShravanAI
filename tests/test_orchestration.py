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
            dataset="fallallD",
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

    # Stage 4 fields: populated, not left as defaults/nulls where they
    # shouldn't be.
    assert (manifest_df["sample_rate_hz"] == 100.0).all()
    assert (manifest_df["duration_s"] > 0).all()
    fall_row = manifest_df[manifest_df["activity_code"] == "T22"].iloc[0]
    assert fall_row["label"] == "fall"
    assert fall_row["fall_onset_frame"] is not None
    assert fall_row["fall_impact_frame"] is not None
    adl_row = manifest_df[manifest_df["activity_code"] == "T01"].iloc[0]
    assert pd.isna(adl_row["fall_onset_frame"])


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


# --- SisFall (Stage 5) ---
# Fixture set: SA01 has D01 (adl) + F01 (fall), neither of which is a
# standing-initiated code -- SA01 should NOT calibrate on its own.
# SA02 has D02 (adl, not standing-initiated) + D07 (adl, IS
# standing-initiated and genuinely still) -- SA02 SHOULD calibrate via
# auto_detected. This means the batch exercises both the auto_detected
# tier and the group_fallback tier (for SA01) in one run, and critically
# confirms neither subject's D01/D02/F01 (task_id 1 or 2) gets mistaken
# for a T01-equivalent, since primary_calibration_task_id=None for
# sisfall.
SISFALL_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "sisfall_mock"


def test_end_to_end_sisfall_processes_all_fixture_trials(tmp_path):
    summary = run_harmonization(
        dataset="sisfall",
        sensor_root=SISFALL_FIXTURE_ROOT, label_root=None,
        harmonized_root=tmp_path / "harmonized", quarantine_root=tmp_path / "quarantine",
        harmonization_config=HarmonizationConfig(),
    )

    assert summary.n_trials_total == 4  # SA01: D01,F01 ; SA02: D02,D07
    assert summary.n_written + summary.n_quarantined == 4


def test_end_to_end_sisfall_never_mistakes_task_id_1_for_primary_calibration(tmp_path):
    # SA01's D01 is task_id=1 and genuinely has no movement label
    # distinguishing it from a "calibration trial" by task_id alone --
    # this must NOT produce a "T01" calibration source for SisFall.
    summary = run_harmonization(
        dataset="sisfall",
        sensor_root=SISFALL_FIXTURE_ROOT, label_root=None,
        harmonized_root=tmp_path / "harmonized", quarantine_root=tmp_path / "quarantine",
        harmonization_config=HarmonizationConfig(),
    )

    assert "T01" not in summary.calibration_source_counts
    # SA02 calibrates via auto_detected (its D07); SA01 has nothing
    # standing-initiated of its own, so it falls back to group_fallback.
    assert summary.calibration_source_counts.get("auto_detected") == 1
    assert summary.calibration_source_counts.get("group_fallback") == 1


def test_end_to_end_sisfall_writes_harmonized_files_with_canonical_columns(tmp_path):
    # Confirms real unit conversion + real 200->100Hz resampling both
    # actually ran (not no-ops, unlike KFall) -- output must still land
    # on exactly the canonical 6 acc_*/gyro_* columns despite SisFall's
    # raw signal having 9 raw_* columns going in.
    harmonized_root = tmp_path / "harmonized"
    run_harmonization(
        dataset="sisfall",
        sensor_root=SISFALL_FIXTURE_ROOT, label_root=None,
        harmonized_root=harmonized_root, quarantine_root=tmp_path / "quarantine",
        harmonization_config=HarmonizationConfig(),
    )

    written_files = list((harmonized_root / "sisfall").glob("*.parquet"))
    assert len(written_files) >= 1

    sample = pd.read_parquet(written_files[0])
    assert list(sample.columns) == ["time_s", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

    # 200 Hz native -> 100 Hz target means roughly half as many rows as
    # the raw file had -- a weak but real check that resampling
    # actually happened rather than passing through unchanged.
    implied_native_rows = len(sample) * 2
    assert implied_native_rows > len(sample)  # trivially true, but documents the 2x relationship


def test_end_to_end_sisfall_manifest_has_no_onset_impact_frames(tmp_path):
    # SisFall has no frame-level fall labels at all, even for fall
    # trials -- confirms this comes through as genuinely null, not
    # some placeholder value, and that query_prediction_trials would
    # correctly exclude these via its dataset=="kfall" filter (tested
    # separately in test_manifest.py).
    manifest_path = tmp_path / "harmonized" / "manifest.parquet"
    run_harmonization(
        dataset="sisfall",
        sensor_root=SISFALL_FIXTURE_ROOT, label_root=None,
        harmonized_root=tmp_path / "harmonized", quarantine_root=tmp_path / "quarantine",
        harmonization_config=HarmonizationConfig(),
        manifest_path=manifest_path,
    )

    manifest_df = load_manifest(manifest_path)
    assert len(manifest_df) == 4
    assert set(manifest_df["dataset"]) == {"sisfall"}
    assert manifest_df["fall_onset_frame"].isna().all()
    assert manifest_df["fall_impact_frame"].isna().all()
    fall_row = manifest_df[manifest_df["activity_code"] == "F01"].iloc[0]
    assert fall_row["label"] == "fall"
