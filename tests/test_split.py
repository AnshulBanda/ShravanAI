"""Tests for detection/split.py."""
import pandas as pd
import pytest

from detection.split import SplitConfig, split_by_subject


def _synthetic_df(n_subjects_per_dataset: int = 10, windows_per_subject: int = 5) -> pd.DataFrame:
    rows = []
    for dataset in ["kfall", "sisfall"]:
        for i in range(n_subjects_per_dataset):
            subject_id = f"S{i:02d}"
            global_subject_id = f"{dataset}_{subject_id}"
            for w in range(windows_per_subject):
                rows.append({
                    "dataset": dataset,
                    "subject_id": subject_id,
                    "global_subject_id": global_subject_id,
                    "label": w % 2,
                    "some_feature": float(w),
                })
    return pd.DataFrame(rows)


def test_split_produces_no_subject_leakage():
    df = _synthetic_df()
    train_df, val_df, test_df = split_by_subject(df, SplitConfig(random_state=0))

    train_subjects = set(train_df["global_subject_id"])
    val_subjects = set(val_df["global_subject_id"])
    test_subjects = set(test_df["global_subject_id"])

    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)


def test_split_covers_every_window_exactly_once():
    df = _synthetic_df()
    train_df, val_df, test_df = split_by_subject(df, SplitConfig(random_state=0))

    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_split_includes_both_datasets_in_every_split():
    df = _synthetic_df(n_subjects_per_dataset=10)
    train_df, val_df, test_df = split_by_subject(df, SplitConfig(random_state=0))

    for split_df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        assert set(split_df["dataset"]) == {"kfall", "sisfall"}, (
            f"{name} split is missing a dataset -- should never happen with "
            f"10 subjects/dataset and default 15%/15% val/test sizes"
        )


def test_split_sizes_roughly_match_requested_fractions():
    # 10 subjects/dataset, 5 windows each = 50 windows/dataset. With
    # test_size=0.2 that's ~2 subjects (~10 windows) held out per
    # dataset -- not exact (subject-level granularity means we can't
    # hit fractions precisely), but should be in a reasonable range.
    df = _synthetic_df(n_subjects_per_dataset=10, windows_per_subject=5)
    train_df, val_df, test_df = split_by_subject(
        df, SplitConfig(val_size=0.2, test_size=0.2, random_state=0)
    )

    total = len(df)
    test_fraction = len(test_df) / total
    val_fraction = len(val_df) / total
    assert 0.05 < test_fraction < 0.4  # generous bounds given subject-level granularity
    assert 0.05 < val_fraction < 0.4


def test_split_is_reproducible_with_same_random_state():
    df = _synthetic_df()
    train_a, val_a, test_a = split_by_subject(df, SplitConfig(random_state=42))
    train_b, val_b, test_b = split_by_subject(df, SplitConfig(random_state=42))

    assert set(train_a["global_subject_id"]) == set(train_b["global_subject_id"])
    assert set(test_a["global_subject_id"]) == set(test_b["global_subject_id"])


def test_split_differs_with_different_random_state():
    df = _synthetic_df(n_subjects_per_dataset=15)
    train_a, _, test_a = split_by_subject(df, SplitConfig(random_state=1))
    train_b, _, test_b = split_by_subject(df, SplitConfig(random_state=2))

    # Not guaranteed to differ in principle, but overwhelmingly likely
    # with 15 subjects/dataset and two different seeds -- a real
    # regression (e.g. random_state silently ignored) would make this
    # fail consistently, not flakily.
    assert set(test_a["global_subject_id"]) != set(test_b["global_subject_id"])


def test_split_raises_clear_error_on_too_few_subjects():
    # Only 2 subjects in a dataset -- can't carve out 3 non-empty splits.
    df = _synthetic_df(n_subjects_per_dataset=2)

    with pytest.raises(ValueError, match="distinct global_subject_id"):
        split_by_subject(df, SplitConfig(random_state=0))


def test_split_handles_single_dataset_input():
    df = _synthetic_df(n_subjects_per_dataset=10)
    df = df[df["dataset"] == "kfall"].reset_index(drop=True)

    train_df, val_df, test_df = split_by_subject(df, SplitConfig(random_state=0))

    assert set(train_df["dataset"]) == {"kfall"}
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
