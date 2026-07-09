"""Cross-dataset harmonization manifest (Stage 4).

The single index both the detection and prediction pipelines will query
against for dataset construction, per the blueprint's §4: one row per
trial, recording enough to filter by dataset/label and to load the
right harmonized file -- without ever persisting anything keyed by a
windowing choice (window size, stride). That stays out-of-scope for
this module by design; windowing happens at dataset-construction time,
downstream, reading THIS manifest.

Schema is a superset of the blueprint's documented columns
(`trial_id, dataset, subject_id, activity_code, label, duration_s,
sample_rate_hz, fall_onset_frame, fall_impact_frame, harmonized_path`)
plus two provenance fields (`accepted`, `calibration_source`) that were
already part of Stage 3's minimal version and are worth keeping --
`accepted` in particular is what lets a query silently exclude
quarantined trials rather than a caller having to know to check.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# Columns that together uniquely identify a trial, used both to key the
# upsert in write_manifest and to dedupe defensively in load_manifest.
_PRIMARY_KEY = ["dataset", "subject_id", "activity_code", "trial_id"]


@dataclass
class ManifestRow:
    dataset: str
    subject_id: str
    activity_code: str
    trial_id: str
    label: str
    duration_s: float
    sample_rate_hz: float
    accepted: bool
    calibration_source: str
    harmonized_path: str
    fall_onset_frame: Optional[int] = None
    fall_impact_frame: Optional[int] = None


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    """Write rows to the manifest at `path`, upserting by primary key.

    This is NOT a plain overwrite. If `path` already has rows (e.g.
    from a previous run on a different dataset, or an earlier run on
    the same dataset before a bugfix), any existing row whose
    (dataset, subject_id, activity_code, trial_id) matches one of the
    new `rows` is replaced; every other existing row is kept untouched.
    This is what makes it safe to run `harmonize_dataset.py --dataset
    kfall` today and `--dataset sisfall` next month without the second
    run silently deleting the first run's rows -- the failure mode the
    original Stage 3 version of this function had (plain
    `df.to_parquet`, unconditional overwrite).

    Re-running the SAME dataset (e.g. after a harmonization bugfix) is
    also safe and intentional: it replaces that dataset's rows with the
    new ones rather than duplicating them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame([asdict(r) for r in rows])

    if path.exists():
        existing_df = pd.read_parquet(path)
        new_keys = set(new_df[_PRIMARY_KEY].itertuples(index=False, name=None))
        existing_keys = list(existing_df[_PRIMARY_KEY].itertuples(index=False, name=None))
        keep_mask = [key not in new_keys for key in existing_keys]
        combined_df = pd.concat(
            [existing_df[keep_mask], new_df], ignore_index=True
        )
    else:
        combined_df = new_df

    combined_df = combined_df.sort_values(_PRIMARY_KEY).reset_index(drop=True)
    combined_df.to_parquet(path, index=False)


def load_manifest(path: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def query_detection_trials(
    manifest_df: pd.DataFrame,
    datasets: Optional[list[str]] = None,
    accepted_only: bool = True,
) -> pd.DataFrame:
    """Rows the detection pipeline should train/eval on.

    Per the blueprint: detection uses every dataset (KFall, SisFall,
    FallAllD), both fall and ADL trials -- it's a binary classifier
    over whole trials/windows, not restricted to any one dataset or
    label. `datasets=None` (default) means "whatever's in the
    manifest", so this doesn't need updating as SisFall/FallAllD get
    added in later stages.
    """
    df = manifest_df
    if datasets is not None:
        df = df[df["dataset"].isin(datasets)]
    if accepted_only:
        df = df[df["accepted"]]
    return df.reset_index(drop=True)


def query_prediction_trials(
    manifest_df: pd.DataFrame,
    accepted_only: bool = True,
) -> pd.DataFrame:
    """Rows the prediction pipeline should train/eval on.

    Per the blueprint: KFall only (it's the only dataset with labeled
    onset/impact frames), and only trials where onset/impact labeling
    actually applies -- fall trials with a labeled onset frame, or ADL
    trials (which have no onset/impact by definition and are used as
    the negative class).
    """
    df = manifest_df[manifest_df["dataset"] == "kfall"]
    df = df[df["fall_onset_frame"].notna() | (df["label"] == "adl")]
    if accepted_only:
        df = df[df["accepted"]]
    return df.reset_index(drop=True)
