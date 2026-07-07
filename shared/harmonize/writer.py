"""Provenance-aware harmonized trial writer.

Writes a harmonized trial's signal to disk as parquet, with a sidecar
JSON file recording provenance (calibration source, filter parameters,
target rate) and validation outcome. Trials with validation issues
(from Task 3.8) are routed to a separate quarantine location instead of
silently joining the accepted data -- the goal is that a "why does this
trial look wrong" investigation six months from now has an immediate,
on-disk answer, not a mystery.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _trial_key(metadata: Any) -> str:
    """Build a stable filename stem from a trial's metadata, duck-typed
    so this works across datasets whose metadata objects may differ.
    """
    subject_id = getattr(metadata, "subject_id", "unknown_subject")
    activity_code = getattr(metadata, "activity_code", "unknown_activity")
    trial_id = getattr(metadata, "trial_id", "unknown_trial")
    return f"{subject_id}_{activity_code}_{trial_id}"


def _build_provenance(
    metadata: Any,
    calibration_source: str,
    issues: list[str],
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    provenance = {
        "dataset": getattr(metadata, "dataset", None),
        "subject_id": getattr(metadata, "subject_id", None),
        "activity_code": getattr(metadata, "activity_code", None),
        "trial_id": getattr(metadata, "trial_id", None),
        "label": getattr(metadata, "label", None),
        "fall_onset_frame": getattr(metadata, "fall_onset_frame", None),
        "fall_impact_frame": getattr(metadata, "fall_impact_frame", None),
        "calibration_source": calibration_source,
        "issues": issues,
        "accepted": len(issues) == 0,
    }
    if extra:
        provenance.update(extra)
    return provenance


def write_harmonized_trial(
    signal: pd.DataFrame,
    metadata: Any,
    calibration_source: str,
    issues: list[str],
    harmonized_root: Path,
    quarantine_root: Path,
    provenance_extra: dict[str, Any] | None = None,
) -> Path:
    """Write a harmonized trial's signal + provenance to disk.

    Trials with no validation issues go under
    `harmonized_root/<dataset>/<trial_key>.parquet`; trials with any
    issues go under `quarantine_root/<dataset>/<trial_key>.parquet`
    instead, with the issues recorded in the sidecar JSON. Returns the
    path the parquet file was actually written to.
    """
    dataset = getattr(metadata, "dataset", "unknown_dataset")
    trial_key = _trial_key(metadata)

    target_root = quarantine_root if issues else harmonized_root
    target_dir = Path(target_root) / dataset
    target_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = target_dir / f"{trial_key}.parquet"
    signal.to_parquet(parquet_path, index=False)

    provenance = _build_provenance(metadata, calibration_source, issues, provenance_extra)
    json_path = parquet_path.with_suffix(".json")
    json_path.write_text(json.dumps(provenance, indent=2, default=str))

    return parquet_path
