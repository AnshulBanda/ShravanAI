"""Tests for prediction/dataset.py -- windows manifest + window loading."""
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from prediction.dataset import CHANNELS, build_windows_manifest, load_window
from prediction.labelers import FALL, LABEL_TO_INT, NON_FALL, PRE_IMPACT
from prediction.windowing import PredictionWindowingConfig
from shared.manifest import ManifestRow


def _trial_row(**overrides) -> dict:
    defaults = dict(
        dataset="kfall", subject_id="SA06", activity_code="T01", trial_id="R01",
        label="adl", duration_s=3.0, sample_rate_hz=100.0, accepted=True,
        calibration_source="T01", harmonized_path="/fake/path.parquet",
        fall_onset_frame=None, fall_impact_frame=None,
    )
    defaults.update(overrides)
    return asdict(ManifestRow(**defaults))


def _dense_config():
    return PredictionWindowingConfig(window_length_s=1.0, stride_s=0.1, target_rate_hz=100.0)


# --- build_windows_manifest ---

def test_adl_trial_all_windows_labeled_non_fall():
    trial_df = pd.DataFrame([_trial_row(label="adl", duration_s=3.0)])

    windows_df = build_windows_manifest(trial_df, config=_dense_config())

    assert len(windows_df) > 0
    assert (windows_df["label"] == NON_FALL).all()
    assert (windows_df["label_id"] == LABEL_TO_INT[NON_FALL]).all()


def test_fall_trial_produces_all_three_label_classes():
    # 400 real samples (4.0s), onset=130, impact=208 -- the project's
    # real SA06 T22 R01 values (per PROJECT_CHECKPOINT.md), so windows
    # well before frame 130, some spanning 130-208, and some at/after
    # 208 should all appear.
    trial_df = pd.DataFrame([_trial_row(
        activity_code="T22", label="fall", duration_s=4.0,
        fall_onset_frame=130, fall_impact_frame=208,
    )])

    windows_df = build_windows_manifest(trial_df, config=_dense_config())

    assert set(windows_df["label"]) == {NON_FALL, PRE_IMPACT, FALL}
    # NOTE: the blueprint's "pre_impact is the rarest class" expectation
    # is a DATASET-level property (driven by ADL trials, which are all
    # non_fall, dominating the pooled window count) -- not guaranteed
    # within one short synthetic fall trial like this one, where the
    # pre-onset segment is deliberately tiny. Real verification happens
    # once real KFall data is run through this. This test only checks
    # all three classes are reachable and correctly ordered in time.
    labels_by_start = windows_df.sort_values("start_frame")["label"].tolist()
    assert labels_by_start[0] == NON_FALL
    assert labels_by_start[-1] == FALL


def test_query_prediction_trials_excludes_non_kfall_dataset():
    trial_df = pd.DataFrame([
        _trial_row(dataset="kfall", activity_code="T01", label="adl", duration_s=2.0),
        _trial_row(dataset="sisfall", subject_id="SA01", activity_code="D01",
                    label="adl", duration_s=2.0),
    ])

    windows_df = build_windows_manifest(trial_df, config=_dense_config())

    assert set(windows_df["dataset"]) == {"kfall"}


def test_fall_trial_with_missing_onset_frame_is_excluded():
    # A fall-labeled trial with no onset frame at all (a labeling gap,
    # per shared/manifest.py's query_prediction_trials docstring) must
    # not silently reach the labeler.
    trial_df = pd.DataFrame([
        _trial_row(activity_code="T22", label="fall", duration_s=2.0,
                    fall_onset_frame=None, fall_impact_frame=None),
    ])

    windows_df = build_windows_manifest(trial_df, config=_dense_config())

    assert len(windows_df) == 0


def test_excludes_unaccepted_trials():
    trial_df = pd.DataFrame([
        _trial_row(activity_code="T01", accepted=True, duration_s=2.0),
        _trial_row(activity_code="T05", accepted=False, duration_s=2.0),
    ])

    windows_df = build_windows_manifest(trial_df, config=_dense_config())

    assert set(windows_df["activity_code"]) == {"T01"}


def test_global_subject_id_disambiguates_kfall_from_other_datasets():
    trial_df = pd.DataFrame([_trial_row(dataset="kfall", subject_id="SA06")])

    windows_df = build_windows_manifest(trial_df, config=_dense_config())

    assert set(windows_df["global_subject_id"]) == {"kfall_SA06"}


def test_empty_input_has_correct_columns_not_just_empty():
    trial_df = pd.DataFrame([_trial_row(accepted=False)])

    windows_df = build_windows_manifest(trial_df, config=_dense_config())

    assert len(windows_df) == 0
    assert "global_subject_id" in windows_df.columns
    assert "label_id" in windows_df.columns


def test_onset_and_impact_frame_carried_through_to_window_level():
    # Every window of a fall trial should carry the SAME onset/impact
    # frame (the source trial's, not per-window) -- needed downstream
    # by prediction.lead_time, which needs the exact impact frame, not
    # an approximation reconstructed from window labels.
    trial_df = pd.DataFrame([_trial_row(
        activity_code="T22", label="fall", duration_s=4.0,
        fall_onset_frame=130, fall_impact_frame=208,
    )])

    windows_df = build_windows_manifest(trial_df, config=_dense_config())

    assert (windows_df["fall_onset_frame"] == 130).all()
    assert (windows_df["fall_impact_frame"] == 208).all()


def test_onset_and_impact_frame_are_none_for_adl_trial():
    trial_df = pd.DataFrame([_trial_row(label="adl", duration_s=2.0)])

    windows_df = build_windows_manifest(trial_df, config=_dense_config())

    assert windows_df["fall_onset_frame"].isna().all()
    assert windows_df["fall_impact_frame"].isna().all()


# --- load_window (same edge-padding contract as detection's) ---

def _write_ramp_parquet(path, n_rows: int):
    df = pd.DataFrame({
        "time_s": np.arange(n_rows) / 100.0,
        **{col: np.arange(n_rows, dtype=np.float32) for col in CHANNELS},
    })
    df.to_parquet(path, index=False)


def test_load_window_extracts_correct_slice(tmp_path):
    path = tmp_path / "trial.parquet"
    _write_ramp_parquet(path, n_rows=300)
    window_row = pd.Series({"harmonized_path": str(path), "start_frame": 190, "end_frame": 290})

    window = load_window(window_row, window_length_samples=100)

    assert window.shape == (100, len(CHANNELS))
    np.testing.assert_array_equal(window[:, 0], np.arange(190, 290))


def test_load_window_pads_short_trailing_window(tmp_path):
    path = tmp_path / "trial.parquet"
    _write_ramp_parquet(path, n_rows=300)
    # Trailing window per test_prediction_windowing.py: 290-300 (10 real samples).
    window_row = pd.Series({"harmonized_path": str(path), "start_frame": 290, "end_frame": 300})

    window = load_window(window_row, window_length_samples=100)

    assert window.shape == (100, len(CHANNELS))
    np.testing.assert_array_equal(window[:10, 0], np.arange(290, 300))
    np.testing.assert_array_equal(window[10:, 0], np.full(90, 299.0))


def test_load_window_zero_real_samples_raises(tmp_path):
    path = tmp_path / "trial.parquet"
    _write_ramp_parquet(path, n_rows=100)
    window_row = pd.Series({"harmonized_path": str(path), "start_frame": 50, "end_frame": 50})

    with pytest.raises(ValueError, match="0 real samples"):
        load_window(window_row, window_length_samples=100)
