"""Tests for shared/io/readers_sisfall.py against synthetic fixture data.

Same approach as test_kfall_reader.py: these fixtures are synthetically
generated to match SisFall's real, Readme-documented format (verified
against actual downloaded SisFall files, not assumed from the paper) --
9 raw ADC columns, comma-separated, semicolon-terminated, no header.
The point is to verify the reader's parsing logic, not to validate
anything about real signal content.
"""
from pathlib import Path

import pytest

from shared.io.readers_sisfall import (
    RAW_COLUMN_ORDER,
    discover_trials,
    load_all_trials,
    load_trial,
    parse_trial_filename,
    read_sensor_txt,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "sisfall_mock"


def test_parse_trial_filename_adl():
    activity_code, subject_id, task_id, trial_id = parse_trial_filename("D01_SA01_R01.txt")
    assert activity_code == "D01"
    assert subject_id == "SA01"
    assert task_id == 1
    assert trial_id == "R01"


def test_parse_trial_filename_fall():
    activity_code, subject_id, task_id, trial_id = parse_trial_filename("F05_SE06_R04.txt")
    assert activity_code == "F05"
    assert subject_id == "SE06"
    assert task_id == 5
    assert trial_id == "R04"


def test_parse_trial_filename_rejects_bad_pattern():
    with pytest.raises(ValueError):
        parse_trial_filename("not_a_sisfall_file.txt")


def test_parse_trial_filename_rejects_readme():
    with pytest.raises(ValueError):
        parse_trial_filename("Readme.txt")


def test_read_sensor_txt_parses_all_rows_and_columns():
    df = read_sensor_txt(FIXTURE_ROOT / "SA01" / "D01_SA01_R01.txt")

    assert list(df.columns) == ["time_s"] + RAW_COLUMN_ORDER
    assert len(df) == 200  # matches the fixture's generated row count


def test_read_sensor_txt_time_column_uses_native_200hz_rate():
    df = read_sensor_txt(FIXTURE_ROOT / "SA01" / "D01_SA01_R01.txt")

    assert df["time_s"].iloc[0] == 0.0
    assert df["time_s"].iloc[1] == pytest.approx(1 / 200.0)


def test_read_sensor_txt_values_are_raw_integers_not_physical_units():
    # The reader must NOT convert units -- that's SisFallUnitConverter's
    # job. A resting-ish ADXL345 z-axis reading should be in the
    # thousands (raw ADC counts), not ~1.0 (g).
    df = read_sensor_txt(FIXTURE_ROOT / "SA01" / "D01_SA01_R01.txt")

    assert df["raw_adxl_acc_z"].mean() > 1000


def test_read_sensor_txt_rejects_wrong_column_count(tmp_path):
    bad_file = tmp_path / "D01_SA99_R01.txt"
    bad_file.write_text("1,2,3,4,5;\n")  # only 5 columns, not 9

    with pytest.raises(ValueError, match="expected 9 columns"):
        read_sensor_txt(bad_file)


def test_load_trial_adl_has_no_fall_labels():
    trial = load_trial(FIXTURE_ROOT / "SA01" / "D01_SA01_R01.txt")

    assert trial.metadata.dataset == "sisfall"
    assert trial.metadata.subject_id == "SA01"
    assert trial.metadata.activity_code == "D01"
    assert trial.metadata.label == "adl"
    assert trial.metadata.fall_onset_frame is None
    assert trial.metadata.fall_impact_frame is None


def test_load_trial_fall_is_labeled_fall_but_still_has_no_frame_labels():
    # SisFall has NO frame-level onset/impact annotation anywhere, even
    # for real fall trials -- this is expected, not a parsing gap. See
    # module docstring.
    trial = load_trial(FIXTURE_ROOT / "SA01" / "F01_SA01_R01.txt")

    assert trial.metadata.label == "fall"
    assert trial.metadata.fall_onset_frame is None
    assert trial.metadata.fall_impact_frame is None


def test_discover_trials_excludes_readme_and_non_matching_files():
    found = discover_trials(FIXTURE_ROOT)
    names = {p.name for p in found}

    assert "Readme.txt" not in names
    assert "notes.txt" not in names
    assert "D01_SA01_R01.txt" in names
    assert "F01_SA01_R01.txt" in names
    assert "D02_SA02_R01.txt" in names


def test_load_all_trials_loads_every_subject():
    trials = load_all_trials(FIXTURE_ROOT)

    assert len(trials) == 3
    assert {t.metadata.subject_id for t in trials} == {"SA01", "SA02"}
    assert {t.metadata.label for t in trials} == {"adl", "fall"}


def test_load_all_trials_accepts_ignored_label_root_for_signature_parity():
    # load_all_trials(sensor_root, label_root) must accept a second
    # positional arg (even though SisFall ignores it) so it can be
    # registered in orchestration._TRIAL_LOADERS alongside KFall's
    # loader without a wrapper function.
    trials = load_all_trials(FIXTURE_ROOT, None)
    assert len(trials) == 3
