"""Subject-aware train/val/test splitting for the detection pipeline.

Splits by SUBJECT, not by window -- a naive random per-window split
would put different windows from the SAME subject's SAME trial (which
share ~50% of their samples, at 50% stride overlap) into different
splits, letting the model implicitly memorize that subject rather than
generalize. This is a well-known leakage failure mode in windowed
sensor-data ML, not a hypothetical concern.

Also splits WITHIN each dataset separately, then combines -- so every
split (train/val/test) is guaranteed to contain both KFall and SisFall
subjects, rather than risking (e.g.) a random subject-level split that
happens to put most of one dataset's subjects into test only. This
matters because the detection pipeline is meant to generalize across
both real-world datasets, not just do well on whichever one dominates
by subject count.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


@dataclass
class SplitConfig:
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42


def split_by_subject(
    df: pd.DataFrame, config: SplitConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split `df` (one row per window or per-window feature vector,
    must have `dataset` and `global_subject_id` columns) into
    (train_df, val_df, test_df).

    No `global_subject_id` ever appears in more than one of the three
    returned frames -- verified as a hard post-condition before
    returning, not just intended.

    Raises ValueError if a dataset doesn't have enough distinct
    subjects to carve out non-empty val/test splits (sklearn's
    GroupShuffleSplit will surface this naturally; re-raised here with
    a clearer message pointing at which dataset is the problem).
    """
    config = config or SplitConfig()

    train_parts, val_parts, test_parts = [], [], []

    for dataset_name, dataset_df in df.groupby("dataset", sort=True):
        dataset_df = dataset_df.reset_index(drop=True)
        n_subjects = dataset_df["global_subject_id"].nunique()
        if n_subjects < 3:
            raise ValueError(
                f"Dataset {dataset_name!r} has only {n_subjects} distinct "
                f"global_subject_id value(s) -- need at least 3 (one per "
                f"split) to carve out a non-empty train/val/test split. "
                f"This is expected on small fixture data; on real data, "
                f"check the input df actually has this dataset's real "
                f"subject variety."
            )

        try:
            test_splitter = GroupShuffleSplit(
                n_splits=1, test_size=config.test_size, random_state=config.random_state
            )
            trainval_idx, test_idx = next(
                test_splitter.split(dataset_df, groups=dataset_df["global_subject_id"])
            )

            trainval_df = dataset_df.iloc[trainval_idx].reset_index(drop=True)
            test_df = dataset_df.iloc[test_idx]

            # val_size was specified as a fraction of the ORIGINAL
            # total, but we're now splitting only the trainval
            # remainder -- rescale so the actual val fraction of the
            # original total still matches what the caller asked for,
            # rather than being val_size of the smaller remainder.
            relative_val_size = config.val_size / (1 - config.test_size)
            val_splitter = GroupShuffleSplit(
                n_splits=1, test_size=relative_val_size, random_state=config.random_state
            )
            train_idx, val_idx = next(
                val_splitter.split(trainval_df, groups=trainval_df["global_subject_id"])
            )
        except ValueError as exc:
            raise ValueError(
                f"Could not split dataset {dataset_name!r} ({n_subjects} subjects) "
                f"into train/val/test with val_size={config.val_size}, "
                f"test_size={config.test_size}: {exc}"
            ) from exc

        train_parts.append(trainval_df.iloc[train_idx])
        val_parts.append(trainval_df.iloc[val_idx])
        test_parts.append(test_df)

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    _assert_no_subject_leakage(train_df, val_df, test_df)
    return train_df, val_df, test_df


def _assert_no_subject_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_subjects = set(train_df["global_subject_id"])
    val_subjects = set(val_df["global_subject_id"])
    test_subjects = set(test_df["global_subject_id"])

    overlaps = {
        "train/val": train_subjects & val_subjects,
        "train/test": train_subjects & test_subjects,
        "val/test": val_subjects & test_subjects,
    }
    leaking = {k: v for k, v in overlaps.items() if v}
    if leaking:
        raise AssertionError(
            f"Subject leakage across splits (should never happen -- this "
            f"is a bug in split_by_subject, not a data issue): {leaking}"
        )
