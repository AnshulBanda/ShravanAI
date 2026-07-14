"""Leave-one-subject-out (LOSO) fold construction for the prediction pipeline.

Per blueprint Pipeline 2 §8: "LOSO within KFall ... remains your
primary evaluation" -- no LODO for this pipeline (KFall-only, by the
"no fabricated labels" design constraint already documented in Stage
7's dataset.py).

This is the mechanism that actually prevents leakage. It's tempting to
conflate this with the "sequence-aware batching" idea from §7 (shuffle
at the trial level when building minibatches) -- they solve DIFFERENT
problems:
  - THIS module (subject-level fold splitting) prevents a held-out test
    subject's windows from ever being seen in training -- this is
    where leakage would actually happen if done wrong, since a random
    per-WINDOW split would put two 90%-overlapping windows from the
    same trial into train and test respectively, silently inflating
    the test score.
  - `torch_dataset.py`'s trial-grouped batch sampler operates AFTER a
    split like this has already happened, and only affects training
    dynamics within one already-clean training set (batch composition,
    not leakage prevention). Keeping this distinction explicit in the
    two modules' docstrings rather than blurring them into one concept,
    since blueprint §7's wording ("to avoid near-duplicate windows
    leaking across train/val") could be misread as this module's job.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LOSOFold:
    test_subject: str            # a global_subject_id, e.g. "kfall_SA06"
    train_subjects: tuple[str, ...]


def generate_loso_folds(windows_df: pd.DataFrame) -> list[LOSOFold]:
    """One fold per unique subject in `windows_df` -- that subject held
    out as test, every other subject as train.

    Subjects are sorted (not just `unique()`'s arbitrary order) so fold
    order is deterministic across runs -- matters for reproducibility
    when reporting per-fold results.
    """
    subjects = sorted(windows_df["global_subject_id"].unique())
    if len(subjects) < 2:
        raise ValueError(
            f"Need at least 2 distinct subjects to build LOSO folds; "
            f"got {len(subjects)}. A single-subject dataset can't have "
            f"a meaningful held-out fold."
        )
    return [
        LOSOFold(test_subject=s, train_subjects=tuple(o for o in subjects if o != s))
        for s in subjects
    ]


def get_fold_masks(windows_df: pd.DataFrame, fold: LOSOFold) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (train_mask, test_mask) arrays for one fold, aligned to
    `windows_df`'s row order. Returned as masks rather than filtered
    copies of `windows_df` since the windows manifest can be large
    (348k+ rows on the real KFall data, per Stage 7's real-data
    verification) -- letting the caller decide whether/how to
    materialize a filtered copy avoids forcing an extra full-DataFrame
    copy per fold when 32+ folds are being run in a loop.
    """
    test_mask = (windows_df["global_subject_id"] == fold.test_subject).to_numpy()
    train_mask = windows_df["global_subject_id"].isin(fold.train_subjects).to_numpy()
    return train_mask, test_mask
