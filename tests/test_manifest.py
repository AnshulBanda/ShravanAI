"""Tests for shared/manifest.py (Stage 3, Task 3.10 + Stage 4 extension)."""
import pandas as pd

from shared.manifest import (
    ManifestRow,
    load_manifest,
    query_detection_trials,
    query_prediction_trials,
    write_manifest,
)


def _row(**overrides) -> ManifestRow:
    """Build a ManifestRow with sensible defaults, overriding as needed."""
    defaults = dict(
        dataset="kfall", subject_id="SA06", activity_code="T01", trial_id="R01",
        label="adl", duration_s=10.0, sample_rate_hz=100.0, accepted=True,
        calibration_source="T01", harmonized_path="/fake/path/SA06_T01_R01.parquet",
        fall_onset_frame=None, fall_impact_frame=None,
    )
    defaults.update(overrides)
    return ManifestRow(**defaults)


def _row_dict(**overrides) -> dict:
    from dataclasses import asdict
    return asdict(_row(**overrides))


def test_write_and_load_manifest_round_trip(tmp_path):
    rows = [
        _row(activity_code="T01", trial_id="R01", label="adl"),
        _row(
            activity_code="T22", trial_id="R01", label="fall",
            fall_onset_frame=130, fall_impact_frame=208,
        ),
    ]
    path = tmp_path / "manifest.parquet"

    write_manifest(rows, path)
    df = load_manifest(path)

    assert len(df) == 2
    assert set(df["subject_id"]) == {"SA06"}
    assert set(df["activity_code"]) == {"T01", "T22"}
    assert df[df["activity_code"] == "T22"]["label"].iloc[0] == "fall"
    assert df[df["activity_code"] == "T22"]["fall_impact_frame"].iloc[0] == 208


def test_write_manifest_creates_parent_directories(tmp_path):
    rows = [_row()]
    path = tmp_path / "nested" / "dir" / "manifest.parquet"

    write_manifest(rows, path)

    assert path.exists()


def test_write_manifest_upserts_same_dataset_without_duplicating(tmp_path):
    """Re-running harmonization on the same dataset (e.g. after a
    bugfix) should REPLACE that dataset's rows, not duplicate them.
    """
    path = tmp_path / "manifest.parquet"

    write_manifest([_row(calibration_source="T01")], path)
    write_manifest([_row(calibration_source="auto_detected")], path)  # re-run, same key

    df = load_manifest(path)
    assert len(df) == 1
    assert df["calibration_source"].iloc[0] == "auto_detected"


def test_write_manifest_preserves_other_datasets(tmp_path):
    """Writing SisFall rows must not wipe out KFall rows already in the
    manifest -- this was a real overwrite bug in the Stage 3 version.
    """
    path = tmp_path / "manifest.parquet"

    write_manifest([_row(dataset="kfall")], path)
    write_manifest([_row(dataset="sisfall", subject_id="SB01")], path)

    df = load_manifest(path)
    assert len(df) == 2
    assert set(df["dataset"]) == {"kfall", "sisfall"}


def test_write_manifest_replaces_only_matching_trials_within_a_dataset(tmp_path):
    """Re-running the same dataset shouldn't clobber OTHER trials in
    that dataset that weren't part of the new write.
    """
    path = tmp_path / "manifest.parquet"

    write_manifest([_row(activity_code="T01"), _row(activity_code="T22", label="fall")], path)
    write_manifest([_row(activity_code="T01", calibration_source="auto_detected")], path)

    df = load_manifest(path)
    assert len(df) == 2  # T22 row untouched, T01 row replaced in place
    assert df[df["activity_code"] == "T01"]["calibration_source"].iloc[0] == "auto_detected"


def test_query_detection_trials_includes_all_datasets_and_labels():
    df = pd.DataFrame([
        _row_dict(dataset="kfall", label="adl"),
        _row_dict(dataset="kfall", activity_code="T22", label="fall"),
        _row_dict(dataset="sisfall", subject_id="SB01", label="fall"),
        _row_dict(dataset="kfall", activity_code="T05", label="adl", accepted=False),
    ])

    result = query_detection_trials(df)

    assert len(result) == 3  # the accepted=False row is excluded
    assert set(result["dataset"]) == {"kfall", "sisfall"}


def test_query_detection_trials_can_filter_datasets():
    df = pd.DataFrame([
        _row_dict(dataset="kfall"),
        _row_dict(dataset="sisfall", subject_id="SB01"),
    ])

    result = query_detection_trials(df, datasets=["kfall"])

    assert set(result["dataset"]) == {"kfall"}


def test_query_prediction_trials_kfall_only():
    df = pd.DataFrame([
        _row_dict(dataset="kfall", label="adl"),
        _row_dict(
            dataset="sisfall", subject_id="SB01", label="fall",
            fall_onset_frame=100, fall_impact_frame=150,
        ),
    ])

    result = query_prediction_trials(df)

    assert set(result["dataset"]) == {"kfall"}


def test_query_prediction_trials_requires_onset_frame_or_adl():
    df = pd.DataFrame([
        _row_dict(activity_code="T01", label="adl", fall_onset_frame=None),
        _row_dict(
            activity_code="T22", label="fall",
            fall_onset_frame=130, fall_impact_frame=208,
        ),
        # A fall-labeled row with no onset frame (e.g. a labeling gap)
        # should be excluded -- the prediction pipeline can't build a
        # pre-impact window without it.
        _row_dict(activity_code="T21", label="fall", fall_onset_frame=None),
    ])

    result = query_prediction_trials(df)

    assert len(result) == 2
    assert set(result["activity_code"]) == {"T01", "T22"}


def test_query_prediction_trials_excludes_unaccepted_by_default():
    df = pd.DataFrame([
        _row_dict(activity_code="T01", label="adl", accepted=True),
        _row_dict(activity_code="T05", label="adl", accepted=False),
    ])

    result = query_prediction_trials(df)

    assert len(result) == 1
    assert result["activity_code"].iloc[0] == "T01"
