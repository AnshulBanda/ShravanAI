"""Tests for detection/dataset.py -- windows manifest + window loading."""
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from detection.dataset import CHANNELS, build_windows_manifest, load_window
from detection.windowing import WindowingConfig
from shared.manifest import ManifestRow


def _trial_row(**overrides) -> dict:
    defaults = dict(
        dataset="kfall", subject_id="SA06", activity_code="T01", trial_id="R01",
        label="adl", duration_s=5.5, sample_rate_hz=100.0, accepted=True,
        calibration_source="T01", harmonized_path="/fake/path.parquet",
        fall_onset_frame=None, fall_impact_frame=None,
    )
    defaults.update(overrides)
    return asdict(ManifestRow(**defaults))


def _small_config():
    # Small window/stride so tests don't need giant fixture trials.
    return WindowingConfig(window_length_s=2.0, stride_s=1.0, target_rate_hz=100.0)


# --- build_windows_manifest ---

def test_build_windows_manifest_basic_counts():
    trial_df = pd.DataFrame([
        _trial_row(activity_code="T01", label="adl", duration_s=5.5),   # 550 samples
    ])

    windows_df = build_windows_manifest(trial_df, config=_small_config())

    # 550 samples @ window=200,stride=100 -> 5 windows (see test_windowing.py)
    assert len(windows_df) == 5
    assert (windows_df["label"] == 0).all()
    assert (windows_df["dataset"] == "kfall").all()


def test_build_windows_manifest_fall_trial_labels_all_windows_as_fall():
    trial_df = pd.DataFrame([
        _trial_row(activity_code="T22", label="fall", duration_s=3.0),
    ])

    windows_df = build_windows_manifest(trial_df, config=_small_config())

    assert len(windows_df) > 0
    assert (windows_df["label"] == 1).all()


def test_build_windows_manifest_excludes_unaccepted_trials():
    trial_df = pd.DataFrame([
        _trial_row(activity_code="T01", accepted=True, duration_s=3.0),
        _trial_row(activity_code="T05", accepted=False, duration_s=3.0),
    ])

    windows_df = build_windows_manifest(trial_df, config=_small_config())

    assert set(windows_df["activity_code"]) == {"T01"}


def test_build_windows_manifest_disambiguates_colliding_subject_ids_across_datasets():
    # KFall and SisFall both use "SA06"-style IDs -- this is a REAL
    # collision in this project's own two datasets, not a hypothetical
    # one. global_subject_id must disambiguate them.
    trial_df = pd.DataFrame([
        _trial_row(dataset="kfall", subject_id="SA06", label="fall", duration_s=2.0),
        _trial_row(dataset="sisfall", subject_id="SA06", label="adl", duration_s=2.0),
    ])

    windows_df = build_windows_manifest(trial_df, config=_small_config())

    assert set(windows_df["subject_id"]) == {"SA06"}  # same raw ID, on purpose
    assert set(windows_df["global_subject_id"]) == {"kfall_SA06", "sisfall_SA06"}
    kfall_windows = windows_df[windows_df["global_subject_id"] == "kfall_SA06"]
    sisfall_windows = windows_df[windows_df["global_subject_id"] == "sisfall_SA06"]
    assert (kfall_windows["label"] == 1).all()
    assert (sisfall_windows["label"] == 0).all()


def test_build_windows_manifest_empty_input_has_correct_columns_not_just_empty():
    trial_df = pd.DataFrame([
        _trial_row(accepted=False),  # nothing survives query_detection_trials
    ])

    windows_df = build_windows_manifest(trial_df, config=_small_config())

    assert len(windows_df) == 0
    assert "global_subject_id" in windows_df.columns
    assert "start_frame" in windows_df.columns


def test_build_windows_manifest_window_indices_are_sequential_per_trial():
    trial_df = pd.DataFrame([_trial_row(duration_s=5.5)])

    windows_df = build_windows_manifest(trial_df, config=_small_config())

    assert list(windows_df["window_index"]) == list(range(len(windows_df)))


# --- load_window ---

def _write_ramp_parquet(path, n_rows: int):
    """A parquet file where each channel's value at row i is simply i
    (as float) -- makes slicing/padding trivially easy to verify by eye.
    """
    df = pd.DataFrame({
        "time_s": np.arange(n_rows) / 100.0,
        **{col: np.arange(n_rows, dtype=np.float32) for col in CHANNELS},
    })
    df.to_parquet(path, index=False)


def test_load_window_extracts_correct_slice(tmp_path):
    path = tmp_path / "trial.parquet"
    _write_ramp_parquet(path, n_rows=300)
    window_row = pd.Series({"harmonized_path": str(path), "start_frame": 50, "end_frame": 100})

    window = load_window(window_row, window_length_samples=50)

    assert window.shape == (50, len(CHANNELS))
    np.testing.assert_array_equal(window[:, 0], np.arange(50, 100))


def test_load_window_pads_short_window_by_repeating_last_real_sample(tmp_path):
    path = tmp_path / "trial.parquet"
    _write_ramp_parquet(path, n_rows=120)
    # Real data available: rows 100-119 (20 real samples), asked for
    # a 50-sample window -> 30 samples of padding needed.
    window_row = pd.Series({"harmonized_path": str(path), "start_frame": 100, "end_frame": 120})

    window = load_window(window_row, window_length_samples=50)

    assert window.shape == (50, len(CHANNELS))
    np.testing.assert_array_equal(window[:20, 0], np.arange(100, 120))
    # Every padded row must equal the LAST real row (edge-padding, not
    # zero-padding) -- last real value here is 119.
    np.testing.assert_array_equal(window[20:, 0], np.full(30, 119.0))


def test_load_window_no_padding_needed_when_exact_length(tmp_path):
    path = tmp_path / "trial.parquet"
    _write_ramp_parquet(path, n_rows=200)
    window_row = pd.Series({"harmonized_path": str(path), "start_frame": 0, "end_frame": 200})

    window = load_window(window_row, window_length_samples=200)

    assert window.shape == (200, len(CHANNELS))
    np.testing.assert_array_equal(window[:, 0], np.arange(200))


def test_load_window_reuses_provided_cache_across_calls(tmp_path):
    path = tmp_path / "trial.parquet"
    _write_ramp_parquet(path, n_rows=300)
    cache: dict = {}
    row_a = pd.Series({"harmonized_path": str(path), "start_frame": 0, "end_frame": 50})
    row_b = pd.Series({"harmonized_path": str(path), "start_frame": 50, "end_frame": 100})

    load_window(row_a, window_length_samples=50, signal_cache=cache)
    assert str(path) in cache
    cached_df_identity = id(cache[str(path)])

    load_window(row_b, window_length_samples=50, signal_cache=cache)
    # Second call for the SAME file must reuse the cached DataFrame,
    # not re-read the parquet -- same object identity in the cache.
    assert id(cache[str(path)]) == cached_df_identity


def test_load_window_zero_real_samples_raises_rather_than_silently_returning_garbage(tmp_path):
    path = tmp_path / "trial.parquet"
    _write_ramp_parquet(path, n_rows=100)
    window_row = pd.Series({"harmonized_path": str(path), "start_frame": 50, "end_frame": 50})

    with pytest.raises(ValueError, match="0 real samples"):
        load_window(window_row, window_length_samples=50)
