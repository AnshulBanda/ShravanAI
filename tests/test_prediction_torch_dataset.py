"""Tests for prediction/torch_dataset.py."""
import numpy as np
import pandas as pd
import pytest
import torch

from prediction.torch_dataset import PredictionWindowDataset, TrialGroupedBatchSampler


def _write_ramp_parquet(path, n_rows: int, offset: float = 0.0):
    df = pd.DataFrame({
        "time_s": np.arange(n_rows) / 100.0,
        "acc_x": np.zeros(n_rows, dtype=np.float32),
        "acc_y": np.zeros(n_rows, dtype=np.float32),
        "acc_z": np.full(n_rows, 1.0 + offset, dtype=np.float32),
        "gyro_x": np.zeros(n_rows, dtype=np.float32),
        "gyro_y": np.zeros(n_rows, dtype=np.float32),
        "gyro_z": np.zeros(n_rows, dtype=np.float32),
    })
    df.to_parquet(path, index=False)


# --- PredictionWindowDataset ---

def test_dataset_item_shape_and_dtype(tmp_path):
    path = tmp_path / "trial.parquet"
    _write_ramp_parquet(path, n_rows=200)
    windows_df = pd.DataFrame([
        {"harmonized_path": str(path), "start_frame": 0, "end_frame": 100, "label_id": 1},
    ])

    dataset = PredictionWindowDataset(windows_df, window_length_samples=100)
    x, y = dataset[0]

    assert x.shape == (9, 100)  # channel-first
    assert x.dtype == torch.float32
    assert y.dtype == torch.long
    assert y.item() == 1


def test_dataset_len_matches_windows_df():
    windows_df = pd.DataFrame([
        {"harmonized_path": "a", "start_frame": 0, "end_frame": 100, "label_id": 0},
        {"harmonized_path": "a", "start_frame": 10, "end_frame": 110, "label_id": 0},
    ])
    dataset = PredictionWindowDataset(windows_df)
    assert len(dataset) == 2


def test_dataset_caches_signal_across_calls(tmp_path):
    path = tmp_path / "trial.parquet"
    _write_ramp_parquet(path, n_rows=200)
    windows_df = pd.DataFrame([
        {"harmonized_path": str(path), "start_frame": 0, "end_frame": 100, "label_id": 0},
        {"harmonized_path": str(path), "start_frame": 10, "end_frame": 110, "label_id": 1},
    ])
    dataset = PredictionWindowDataset(windows_df, window_length_samples=100)

    dataset[0]
    assert str(path) in dataset._signal_cache
    cached_df = dataset._signal_cache[str(path)]

    dataset[1]
    # Same object, not re-read/replaced.
    assert dataset._signal_cache[str(path)] is cached_df


# --- TrialGroupedBatchSampler ---

def _multi_trial_windows_df():
    # 3 "trials" (paths), different window counts, to exercise batches
    # spanning group boundaries.
    rows = []
    for path, n_windows in [("trialA", 5), ("trialB", 3), ("trialC", 4)]:
        for i in range(n_windows):
            rows.append({"harmonized_path": path, "start_frame": i * 10, "end_frame": i * 10 + 100})
    return pd.DataFrame(rows)


def test_batch_sampler_covers_every_index_exactly_once():
    windows_df = _multi_trial_windows_df()
    sampler = TrialGroupedBatchSampler(windows_df, batch_size=4, shuffle=True, seed=0)

    all_yielded = [i for batch in sampler for i in batch]

    assert sorted(all_yielded) == list(range(len(windows_df)))


def test_batch_sampler_preserves_within_trial_order():
    windows_df = _multi_trial_windows_df()
    sampler = TrialGroupedBatchSampler(windows_df, batch_size=100, shuffle=False, seed=0)
    # batch_size=100 with shuffle=False -> single batch containing
    # every index in original DataFrame order (group order 0,1,2 when
    # unshuffled), so it should equal a simple concatenation.
    (batch,) = list(sampler)

    trial_a_idx = windows_df.index[windows_df["harmonized_path"] == "trialA"].tolist()
    trial_b_idx = windows_df.index[windows_df["harmonized_path"] == "trialB"].tolist()
    trial_c_idx = windows_df.index[windows_df["harmonized_path"] == "trialC"].tolist()
    assert batch == trial_a_idx + trial_b_idx + trial_c_idx


def test_batch_sampler_len_matches_expected_batch_count():
    windows_df = _multi_trial_windows_df()  # 12 total windows
    sampler = TrialGroupedBatchSampler(windows_df, batch_size=5)
    assert len(sampler) == 3  # ceil(12/5)
    assert len(list(sampler)) == 3


def test_shuffle_true_changes_group_order_across_seeds():
    windows_df = _multi_trial_windows_df()
    sampler_a = TrialGroupedBatchSampler(windows_df, batch_size=100, shuffle=True, seed=1)
    sampler_b = TrialGroupedBatchSampler(windows_df, batch_size=100, shuffle=True, seed=2)

    (batch_a,) = list(sampler_a)
    (batch_b,) = list(sampler_b)

    assert batch_a != batch_b  # different seeds -> different group order (extremely unlikely to collide with only 3 groups... verified below to be deterministic per-seed instead)


def test_shuffle_deterministic_given_same_seed():
    windows_df = _multi_trial_windows_df()
    sampler_1 = TrialGroupedBatchSampler(windows_df, batch_size=100, shuffle=True, seed=42)
    sampler_2 = TrialGroupedBatchSampler(windows_df, batch_size=100, shuffle=True, seed=42)

    (batch_1,) = list(sampler_1)
    (batch_2,) = list(sampler_2)

    assert batch_1 == batch_2
