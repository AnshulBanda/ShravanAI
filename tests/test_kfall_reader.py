"""Tests for shared/io/readers_kfall.py against synthetic fixture data.

These fixtures are NOT real KFall files -- they're synthetically generated
to match KFall's documented column schema and filename convention. The
point is to verify the reader's parsing logic (filename parsing, column
handling, label cross-referencing, fall/ADL labeling) is correct, not to
validate anything about real KFall signal content. Once real KFall files
are available, add a small `test_kfall_reader_real.py` that spot-checks
a handful of real trials the same way -- keep this file as the fast,
no-external-data regression test.
"""
from pathlib import Path

import pytest

from shared.io.readers_kfall import (
    FALL_TASK_IDS,
    discover_trials,
    load_all_trials,
    load_trial,
    parse_trial_filename,
    read_label_file,
    read_sensor_csv,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "kfall_mock"
SENSOR_ROOT = FIXTURE_ROOT / "sensor_data"
LABEL_ROOT = FIXTURE_ROOT / "label_data"


def test_parse_trial_filename():
    subject_id, task_id, trial_id = parse_trial_filename("SA06T22R01.csv")
    assert subject_id == "SA06"
    assert task_id == 22
    assert trial_id == "R01"


def test_parse_trial_filename_rejects_bad_pattern():
    with pytest.raises(ValueError):
        parse_trial_filename("not_a_kfall_file.csv")


def test_fall_task_ids_boundaries():
    assert 22 in FALL_TASK_IDS
    assert 36 in FALL_TASK_IDS
    assert 21 not in FALL_TASK_IDS
    assert 37 not in FALL_TASK_IDS
    assert 5 not in FALL_TASK_IDS


def test_discover_trials_finds_both_fixture_files():
    found = discover_trials(SENSOR_ROOT)
    names = {p.name for p in found}
    assert names == {"SA06T05R01.csv", "SA06T22R01.csv"}


def test_read_sensor_csv_shape_and_columns():
    df = read_sensor_csv(SENSOR_ROOT / "SA06" / "SA06T05R01.csv")
    assert len(df) == 300
    expected_cols = {
        "time_s", "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z",
        "euler_x", "euler_y", "euler_z",
    }
    assert set(df.columns) == expected_cols
    # stand-still-ish synthetic ADL trial should hover near 1g on the
    # vertical (z) axis before any harmonization/alignment is applied
    assert abs(df["acc_z"].mean() - 1.0) < 0.1


def test_read_label_file_normalizes_columns():
    label_df = read_label_file(LABEL_ROOT / "SA06_label.xlsx")
    assert "fall_onset_frame" in label_df.columns
    assert "fall_impact_frame" in label_df.columns
    assert "task_code" in label_df.columns


def test_load_trial_adl_has_no_onset_impact():
    label_df = read_label_file(LABEL_ROOT / "SA06_label.xlsx")
    trial = load_trial(SENSOR_ROOT / "SA06" / "SA06T05R01.csv", label_df)
    assert trial.metadata.label == "adl"
    assert trial.metadata.fall_onset_frame is None
    assert trial.metadata.fall_impact_frame is None
    assert trial.metadata.subject_id == "SA06"
    assert trial.metadata.task_id == 5


def test_load_trial_fall_has_onset_impact_from_label_sheet():
    label_df = read_label_file(LABEL_ROOT / "SA06_label.xlsx")
    trial = load_trial(SENSOR_ROOT / "SA06" / "SA06T22R01.csv", label_df)
    assert trial.metadata.label == "fall"
    assert trial.metadata.fall_onset_frame == 140
    assert trial.metadata.fall_impact_frame == 158
    assert trial.metadata.trial_id == "R01"


def test_load_trial_without_label_df_still_works():
    # ADL trials, and any trial when no label file exists for a subject,
    # should still parse successfully with onset/impact left as None.
    trial = load_trial(SENSOR_ROOT / "SA06" / "SA06T05R01.csv", label_df=None)
    assert trial.metadata.fall_onset_frame is None
    assert trial.signal.shape[0] == 300


def test_load_all_trials_end_to_end():
    trials = load_all_trials(SENSOR_ROOT, LABEL_ROOT)
    assert len(trials) == 2

    by_task = {t.metadata.task_id: t for t in trials}
    assert by_task[5].metadata.label == "adl"
    assert by_task[22].metadata.label == "fall"
    assert by_task[22].metadata.fall_onset_frame == 140
    assert by_task[22].metadata.fall_impact_frame == 158
