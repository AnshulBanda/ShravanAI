"""Minimal harmonization manifest.

Records one row per trial with its harmonization outcome. This is
deliberately minimal -- just enough for Task 3.10's end-to-end script to
report what happened. The fuller cross-dataset manifest (the single
index both the detection and prediction pipelines will query against
for dataset construction) is a later stage once SisFall/FallAllD exist
too; this module's schema is a subset of what that will eventually need,
not a replacement for it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ManifestRow:
    dataset: str
    subject_id: str
    activity_code: str
    trial_id: str
    label: str
    accepted: bool
    calibration_source: str
    harmonized_path: str


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_parquet(path, index=False)


def load_manifest(path: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))
