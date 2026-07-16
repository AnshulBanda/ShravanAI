"""Training loop core for the prediction pipeline's model branches.

Reusable across BOTH `ConvLSTM` and `TinyTransformer` (per blueprint
§6's ablation plan -- "run ConvLSTM and the tiny-Transformer as the
two branches, and decide via ablation") -- this module doesn't import
either model directly; callers pass in an already-constructed
`nn.Module`, so the same `train_one_fold()` works for either without
duplicating the training loop per architecture.

One thing this module does NOT do: run all 32 LOSO folds. That's
`scripts/train_prediction_model.py`'s job (a thin loop calling
`train_one_fold` once per fold from `prediction.loso.generate_loso_
folds`) -- kept separate so the expensive, real-hardware-dependent
part (actually running 32 folds x however many epochs x 2 candidate
architectures) is a script you run and control from the command line,
not something baked into an importable, unit-testable module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from prediction.loso import LOSOFold, get_fold_masks
from prediction.torch_dataset import PredictionWindowDataset, TrialGroupedBatchSampler


@dataclass
class TrainingConfig:
    max_epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    early_stopping_patience: int = 5   # epochs with no val-loss improvement before stopping
    val_fraction: float = 0.15         # fraction of a fold's TRAIN subjects held out for early-stopping validation (never the LOSO test subject itself)
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class EpochHistory:
    train_loss: list[float]
    val_loss: list[float]


@dataclass
class FoldResult:
    fold: LOSOFold
    best_val_loss: float
    best_epoch: int              # 1-indexed epoch at which best_val_loss occurred
    history: EpochHistory
    test_predicted_label_ids: np.ndarray   # argmax predictions on the held-out test subject's windows, same row order as test_windows_df
    test_true_label_ids: np.ndarray
    test_windows_df: pd.DataFrame          # the held-out subject's window rows, for downstream lead-time analysis (needs start_frame/trial identity, not just labels)


def split_train_val_subjects(
    train_subjects: tuple[str, ...],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Randomly hold out `val_fraction` of a LOSO fold's TRAIN subjects
    for early-stopping validation.

    This is deliberately a SEPARATE, subject-level split from the LOSO
    test subject -- the val subjects here are still ordinary training-
    pool subjects (never the fold's held-out test subject), used only
    to decide when to stop training / which epoch's weights to keep.
    Subject-level (not window-level) for the same leakage reason
    `prediction.loso` and `detection.split` already document: windows
    from one subject's trials overlap heavily, so a window-level split
    would leak near-duplicate windows across train/val.

    Always keeps at least 1 subject in each of train/val (raises if
    that's not possible, rather than silently returning an empty val
    set that would make early stopping meaningless).
    """
    subjects = sorted(train_subjects)  # deterministic input order before shuffling
    if len(subjects) < 2:
        raise ValueError(
            f"Need at least 2 train subjects to carve out a val subject; "
            f"got {len(subjects)}."
        )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(subjects).tolist()

    n_val = max(1, round(len(shuffled) * val_fraction))
    n_val = min(n_val, len(shuffled) - 1)  # always leave >=1 train subject

    val_subjects = sorted(shuffled[:n_val])
    train_only_subjects = sorted(shuffled[n_val:])
    return train_only_subjects, val_subjects


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> float:
    """Run one pass over `loader`. Training mode (with backward +
    optimizer step) if `optimizer` is given; eval mode (no grad) if
    `optimizer` is None -- one function for both train and val epochs,
    rather than two near-duplicate loops, since the only real
    difference is whether gradients get computed and applied.

    Returns the loss averaged over SAMPLES, not batches -- batches are
    generally uneven in size here (dense-stride windowing + a trailing
    partial batch), so a naive per-batch average would silently
    over-weight the smaller trailing batch.
    """
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_samples = 0
    with torch.set_grad_enabled(is_training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            total_samples += len(y)

    if total_samples == 0:
        raise ValueError("Empty loader -- 0 samples processed in this epoch.")
    return total_loss / total_samples


def train_one_fold(
    windows_df: pd.DataFrame,
    fold: LOSOFold,
    model: nn.Module,
    loss_fn: nn.Module,
    config: Optional[TrainingConfig] = None,
    on_epoch_end: Optional[Callable[[int, float, float, float], None]] = None,
) -> FoldResult:
    """Train `model` (already constructed -- either `ConvLSTM()` or
    `TinyTransformer()`, or any other module matching their (batch, 9,
    100) -> (batch, 3) logits contract) on one LOSO fold, with early
    stopping, then evaluate on the held-out test subject.

    `model` is mutated in place (its weights end up at the BEST epoch's
    state, not necessarily the last epoch's -- restored explicitly
    below after early stopping, not just left at whatever the final
    epoch produced).

    `on_epoch_end`, if given, is called after EVERY epoch (not just
    improvements) as `on_epoch_end(epoch, train_loss, val_loss,
    epoch_seconds)` -- added specifically so a caller (the CLI script)
    can print live per-epoch progress. Without this, a single fold can
    silently run for many minutes with zero output, which is
    indistinguishable from a hang -- a real, reported point of
    confusion during the first real training run on this pipeline (see
    PROJECT_CHECKPOINT.md). Kept as an optional callback rather than
    hardcoding a `print()` here, since this module is also used
    directly by `tests/test_prediction_training.py`, where per-epoch
    print spam during the test suite would be unwanted noise.
    """
    config = config or TrainingConfig()
    model = model.to(config.device)

    train_mask, test_mask = get_fold_masks(windows_df, fold)
    fold_train_df = windows_df[train_mask].reset_index(drop=True)
    test_df = windows_df[test_mask].reset_index(drop=True)

    train_only_subjects, val_subjects = split_train_val_subjects(
        fold.train_subjects, val_fraction=config.val_fraction, seed=config.seed
    )
    train_df = fold_train_df[fold_train_df["global_subject_id"].isin(train_only_subjects)].reset_index(drop=True)
    val_df = fold_train_df[fold_train_df["global_subject_id"].isin(val_subjects)].reset_index(drop=True)

    train_loader = DataLoader(
        PredictionWindowDataset(train_df),
        batch_sampler=TrialGroupedBatchSampler(train_df, config.batch_size, shuffle=True, seed=config.seed),
    )
    # Val/test: no need for trial-grouped shuffling (that's a training-
    # dynamics concern, per torch_dataset.py's docstring) -- plain
    # sequential batching is fine and simpler here.
    val_loader = DataLoader(PredictionWindowDataset(val_df), batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(PredictionWindowDataset(test_df), batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history = EpochHistory(train_loss=[], val_loss=[])
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    epochs_since_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.monotonic()
        train_loss = run_epoch(model, train_loader, loss_fn, config.device, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn, config.device, optimizer=None)
        epoch_seconds = time.monotonic() - epoch_start
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)

        if on_epoch_end is not None:
            on_epoch_end(epoch, train_loss, val_loss, epoch_seconds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= config.early_stopping_patience:
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    predicted_ids, true_ids = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(config.device))
            predicted_ids.append(logits.argmax(dim=1).cpu().numpy())
            true_ids.append(y.numpy())

    return FoldResult(
        fold=fold,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        history=history,
        test_predicted_label_ids=np.concatenate(predicted_ids) if predicted_ids else np.array([], dtype=int),
        test_true_label_ids=np.concatenate(true_ids) if true_ids else np.array([], dtype=int),
        test_windows_df=test_df,
    )
