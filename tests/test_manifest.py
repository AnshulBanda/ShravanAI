"""Tests for shared/manifest.py (part of Stage 3, Task 3.10)."""
from shared.manifest import ManifestRow, load_manifest, write_manifest


def test_write_and_load_manifest_round_trip(tmp_path):
    rows = [
        ManifestRow(
            dataset="kfall", subject_id="SA06", activity_code="T01", trial_id="R01",
            label="adl", accepted=True, calibration_source="T01",
            harmonized_path="/fake/path/SA06_T01_R01.parquet",
        ),
        ManifestRow(
            dataset="kfall", subject_id="SA06", activity_code="T22", trial_id="R01",
            label="fall", accepted=True, calibration_source="T01",
            harmonized_path="/fake/path/SA06_T22_R01.parquet",
        ),
    ]
    path = tmp_path / "manifest.parquet"

    write_manifest(rows, path)
    df = load_manifest(path)

    assert len(df) == 2
    assert set(df["subject_id"]) == {"SA06"}
    assert set(df["activity_code"]) == {"T01", "T22"}
    assert df[df["activity_code"] == "T22"]["label"].iloc[0] == "fall"


def test_write_manifest_creates_parent_directories(tmp_path):
    rows = [ManifestRow(
        dataset="kfall", subject_id="SA06", activity_code="T01", trial_id="R01",
        label="adl", accepted=True, calibration_source="T01",
        harmonized_path="/fake/path.parquet",
    )]
    path = tmp_path / "nested" / "dir" / "manifest.parquet"

    write_manifest(rows, path)

    assert path.exists()
