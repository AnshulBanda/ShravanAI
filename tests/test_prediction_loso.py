"""Tests for prediction/loso.py -- LOSO fold construction."""
import numpy as np
import pandas as pd
import pytest

from prediction.loso import generate_loso_folds, get_fold_masks


def _windows_df(subjects_and_counts: dict[str, int]) -> pd.DataFrame:
    rows = []
    for subject, count in subjects_and_counts.items():
        rows.extend([{"global_subject_id": subject, "window_index": i} for i in range(count)])
    return pd.DataFrame(rows)


def test_one_fold_per_subject():
    windows_df = _windows_df({"kfall_SA01": 10, "kfall_SA02": 5, "kfall_SA03": 8})

    folds = generate_loso_folds(windows_df)

    assert len(folds) == 3
    assert {f.test_subject for f in folds} == {"kfall_SA01", "kfall_SA02", "kfall_SA03"}


def test_test_subject_excluded_from_its_own_train_subjects():
    windows_df = _windows_df({"kfall_SA01": 10, "kfall_SA02": 5, "kfall_SA03": 8})

    folds = generate_loso_folds(windows_df)

    for fold in folds:
        assert fold.test_subject not in fold.train_subjects
        assert len(fold.train_subjects) == 2


def test_folds_are_deterministically_ordered():
    windows_df = _windows_df({"kfall_SA03": 1, "kfall_SA01": 1, "kfall_SA02": 1})

    folds = generate_loso_folds(windows_df)

    assert [f.test_subject for f in folds] == ["kfall_SA01", "kfall_SA02", "kfall_SA03"]


def test_single_subject_raises():
    windows_df = _windows_df({"kfall_SA01": 10})

    with pytest.raises(ValueError, match="at least 2"):
        generate_loso_folds(windows_df)


def test_fold_masks_partition_disjoint_and_complete():
    windows_df = _windows_df({"kfall_SA01": 10, "kfall_SA02": 5, "kfall_SA03": 8})
    fold = generate_loso_folds(windows_df)[0]  # test_subject = kfall_SA01

    train_mask, test_mask = get_fold_masks(windows_df, fold)

    assert test_mask.sum() == 10
    assert train_mask.sum() == 13  # 5 + 8
    # No row is ever in both -- the actual leakage-prevention property.
    assert not np.any(train_mask & test_mask)
    # Every row accounted for.
    assert np.all(train_mask | test_mask)


def test_fold_masks_correct_subject_assignment():
    windows_df = _windows_df({"kfall_SA01": 3, "kfall_SA02": 2})
    fold = next(f for f in generate_loso_folds(windows_df) if f.test_subject == "kfall_SA01")

    train_mask, test_mask = get_fold_masks(windows_df, fold)

    assert set(windows_df.loc[test_mask, "global_subject_id"]) == {"kfall_SA01"}
    assert set(windows_df.loc[train_mask, "global_subject_id"]) == {"kfall_SA02"}
