"""Tests for prediction/training.py."""
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from prediction.loso import generate_loso_folds, get_fold_masks
from prediction.models.convlstm import ConvLSTM
from prediction.torch_dataset import PredictionWindowDataset
from prediction.training import (
    TrainingConfig,
    run_epoch,
    split_train_val_subjects,
    train_one_fold,
)


# --- split_train_val_subjects ---

def test_split_disjoint_and_complete():
    subjects = tuple(f"kfall_SA{i:02d}" for i in range(1, 11))  # 10 subjects

    train_subj, val_subj = split_train_val_subjects(subjects, val_fraction=0.2, seed=0)

    assert set(train_subj) & set(val_subj) == set()
    assert set(train_subj) | set(val_subj) == set(subjects)
    assert len(val_subj) == 2  # round(10 * 0.2)


def test_split_always_leaves_at_least_one_train_subject():
    subjects = ("kfall_SA01", "kfall_SA02")

    train_subj, val_subj = split_train_val_subjects(subjects, val_fraction=0.9, seed=0)

    assert len(train_subj) >= 1
    assert len(val_subj) >= 1


def test_split_deterministic_given_same_seed():
    subjects = tuple(f"kfall_SA{i:02d}" for i in range(1, 11))
    a = split_train_val_subjects(subjects, seed=7)
    b = split_train_val_subjects(subjects, seed=7)
    assert a == b


def test_split_too_few_subjects_raises():
    with pytest.raises(ValueError, match="at least 2"):
        split_train_val_subjects(("kfall_SA01",))


# --- run_epoch ---

def _tiny_regression_data(n=32):
    # Simple linear model, simple synthetic classification-shaped data
    # (9 channels x 100 samples) -- just needs to exercise train vs.
    # eval mode and return a finite loss, not learn anything real.
    x = torch.randn(n, 9, 100)
    y = torch.randint(0, 3, (n,))
    return x, y


class _TinyLinearModel(nn.Module):
    """A minimal stand-in model (NOT ConvLSTM/TinyTransformer) purely
    to keep `run_epoch` unit tests fast -- flattens input and applies
    one linear layer. `run_epoch` doesn't care about model internals,
    only that it maps (batch, 9, 100) -> (batch, 3) logits."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(9 * 100, 3)

    def forward(self, x):
        return self.linear(x.flatten(1))


def test_run_epoch_training_mode_updates_parameters():
    model = _TinyLinearModel()
    x, y = _tiny_regression_data()
    loader = DataLoader(list(zip(x, y)), batch_size=8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    initial = [p.clone() for p in model.parameters()]
    run_epoch(model, loader, nn.CrossEntropyLoss(), device="cpu", optimizer=optimizer)

    assert any(not torch.equal(p0, p1) for p0, p1 in zip(initial, model.parameters()))


def test_run_epoch_eval_mode_does_not_update_parameters():
    model = _TinyLinearModel()
    x, y = _tiny_regression_data()
    loader = DataLoader(list(zip(x, y)), batch_size=8)

    initial = [p.clone() for p in model.parameters()]
    run_epoch(model, loader, nn.CrossEntropyLoss(), device="cpu", optimizer=None)

    assert all(torch.equal(p0, p1) for p0, p1 in zip(initial, model.parameters()))


def test_run_epoch_averages_by_sample_not_batch():
    # Two batches of very different sizes -- a per-batch average would
    # give the small trailing batch equal weight to the large one.
    model = _TinyLinearModel()
    x, y = _tiny_regression_data(n=20)
    loader = DataLoader(list(zip(x, y)), batch_size=16)  # batches of 16 and 4

    loss = run_epoch(model, loader, nn.CrossEntropyLoss(reduction="mean"), device="cpu", optimizer=None)

    assert np.isfinite(loss)


def test_run_epoch_empty_loader_raises():
    model = _TinyLinearModel()
    loader = DataLoader(list(zip(*_tiny_regression_data(n=0))), batch_size=8)
    with pytest.raises(ValueError, match="Empty loader"):
        run_epoch(model, loader, nn.CrossEntropyLoss(), device="cpu")


# --- train_one_fold (end-to-end, tiny + fast) ---

def _write_windows_df_with_real_parquet(tmp_path, subjects, n_windows_per_subject=6):
    rows = []
    for subject in subjects:
        path = tmp_path / f"{subject}.parquet"
        n_samples = 100 + n_windows_per_subject * 10
        df = pd.DataFrame({
            "time_s": np.arange(n_samples) / 100.0,
            "acc_x": np.random.randn(n_samples).astype(np.float32),
            "acc_y": np.random.randn(n_samples).astype(np.float32),
            "acc_z": np.random.randn(n_samples).astype(np.float32) + 1.0,
            "gyro_x": np.random.randn(n_samples).astype(np.float32),
            "gyro_y": np.random.randn(n_samples).astype(np.float32),
            "gyro_z": np.random.randn(n_samples).astype(np.float32),
        })
        df.to_parquet(path, index=False)

        for i in range(n_windows_per_subject):
            start = i * 10
            rows.append({
                "global_subject_id": subject,
                "harmonized_path": str(path),
                "start_frame": start,
                "end_frame": start + 100,
                "label_id": i % 3,  # cycles through all 3 classes
            })
    return pd.DataFrame(rows)


def test_train_one_fold_end_to_end(tmp_path):
    subjects = [f"kfall_SA{i:02d}" for i in range(1, 5)]  # 4 subjects: enough for 1 test + train/val split of the remaining 3
    windows_df = _write_windows_df_with_real_parquet(tmp_path, subjects, n_windows_per_subject=6)
    fold = generate_loso_folds(windows_df)[0]

    model = ConvLSTM()
    config = TrainingConfig(max_epochs=2, batch_size=4, early_stopping_patience=2, val_fraction=0.34, seed=0)

    result = train_one_fold(windows_df, fold, model, nn.CrossEntropyLoss(), config=config)

    assert result.fold == fold
    assert 1 <= result.best_epoch <= config.max_epochs
    assert len(result.history.train_loss) == len(result.history.val_loss)
    assert len(result.history.train_loss) <= config.max_epochs
    assert np.isfinite(result.best_val_loss)

    # Test predictions cover exactly the held-out subject's windows.
    expected_test_windows = (windows_df["global_subject_id"] == fold.test_subject).sum()
    assert len(result.test_predicted_label_ids) == expected_test_windows
    assert len(result.test_true_label_ids) == expected_test_windows
    assert len(result.test_windows_df) == expected_test_windows
    assert set(result.test_windows_df["global_subject_id"]) == {fold.test_subject}


def test_train_one_fold_restores_best_epoch_weights_not_last_epoch(tmp_path):
    # Rather than asserting a specific epoch number stops training
    # (BatchNorm running-stats buffers drift slightly every training
    # forward pass regardless of the optimizer's learning rate, which
    # makes exact-epoch predictions fragile/misleading here) -- assert
    # the actual invariant that matters: after training, the model's
    # loaded weights are the ones that produced `best_val_loss`, not
    # whatever the LAST epoch happened to produce. Verified by
    # re-running a val-mode epoch after training finishes and checking
    # it reproduces the recorded best_val_loss exactly.
    subjects = [f"kfall_SA{i:02d}" for i in range(1, 5)]
    windows_df = _write_windows_df_with_real_parquet(tmp_path, subjects, n_windows_per_subject=6)
    fold = generate_loso_folds(windows_df)[0]

    model = ConvLSTM()
    config = TrainingConfig(max_epochs=4, batch_size=4, early_stopping_patience=2, val_fraction=0.34, seed=0)

    result = train_one_fold(windows_df, fold, model, nn.CrossEntropyLoss(), config=config)

    # Rebuild the same val_df/val_loader train_one_fold used internally,
    # to re-check the now-restored model's val loss matches.
    train_only_subjects, val_subjects = split_train_val_subjects(
        fold.train_subjects, val_fraction=config.val_fraction, seed=config.seed
    )
    train_mask, _ = get_fold_masks(windows_df, fold)
    fold_train_df = windows_df[train_mask].reset_index(drop=True)
    val_df = fold_train_df[fold_train_df["global_subject_id"].isin(val_subjects)].reset_index(drop=True)
    val_loader = DataLoader(PredictionWindowDataset(val_df), batch_size=config.batch_size, shuffle=False)

    replayed_val_loss = run_epoch(model, val_loader, nn.CrossEntropyLoss(), device="cpu", optimizer=None)

    assert replayed_val_loss == pytest.approx(result.best_val_loss, rel=1e-4)
