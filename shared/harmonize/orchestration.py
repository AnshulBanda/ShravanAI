"""End-to-end per-dataset harmonization orchestration.

Ties together everything from Stage 2 (readers) and Stage 3 (units,
resampling, alignment, filtering, validation, writing) into one
callable: load all trials for a dataset, calibrate every subject
(two-pass: per-subject first, group-average fallback second), then
harmonize/validate/write every trial, finishing with a summary.

Kept in `shared/` rather than directly in `scripts/harmonize_dataset.py`
(where the Stage 3 sprint plan originally placed it) to stay consistent
with this repo's existing convention that `scripts/` holds thin CLI
entry points only, no logic -- see PROJECT_CHECKPOINT.md / blueprint.
The script just parses args, resolves paths from config, and calls
`run_harmonization`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from shared.harmonize.axis_alignment import (
    STANDING_INITIATED_TASK_IDS,
    calibrate_subject,
    resolve_group_fallback,
    summarize_calibration_sources,
)
from shared.harmonize.pipeline import HarmonizationConfig, harmonize_trial
from shared.harmonize.validation import validate_harmonized_trial
from shared.harmonize.writer import write_harmonized_trial
from shared.io.readers_kfall import load_all_trials as _load_kfall_trials
from shared.manifest import ManifestRow, write_manifest

# Registry of per-dataset trial loaders. Only KFall exists as of Stage
# 3 -- SisFall/FallAllD get added here once their readers exist, same
# pattern as units.py's converter registry.
_TRIAL_LOADERS = {
    "kfall": _load_kfall_trials,
}


@dataclass
class HarmonizationSummary:
    n_trials_total: int
    n_written: int
    n_quarantined: int
    calibration_source_counts: dict[str, int]


def run_harmonization(
    dataset: str,
    sensor_root: Path,
    label_root: Path,
    harmonized_root: Path,
    quarantine_root: Path,
    harmonization_config: HarmonizationConfig,
    manifest_path: Optional[Path] = None,
) -> HarmonizationSummary:
    """Harmonize every trial in `dataset`, end to end.

    Raises ValueError if `dataset` has no registered trial loader.
    """
    if dataset not in _TRIAL_LOADERS:
        raise ValueError(
            f"No trial loader registered for dataset {dataset!r}. "
            f"Known datasets: {sorted(_TRIAL_LOADERS)}"
        )

    trials = _TRIAL_LOADERS[dataset](sensor_root, label_root)

    by_subject: dict[str, list] = {}
    for trial in trials:
        by_subject.setdefault(trial.metadata.subject_id, []).append(trial)

    per_subject_calibration = {
        subject_id: calibrate_subject(subject_trials, STANDING_INITIATED_TASK_IDS)
        for subject_id, subject_trials in by_subject.items()
    }
    resolved_calibrations = resolve_group_fallback(per_subject_calibration)

    manifest_rows: list[ManifestRow] = []
    n_written = 0
    n_quarantined = 0

    for trial in trials:
        calibration = resolved_calibrations[trial.metadata.subject_id]

        harmonized_signal = harmonize_trial(trial, calibration, harmonization_config)
        issues = validate_harmonized_trial(
            harmonized_signal,
            trial.metadata,
            calibration,
            expected_rate_hz=harmonization_config.target_rate_hz,
        )
        written_path = write_harmonized_trial(
            harmonized_signal,
            trial.metadata,
            calibration.source,
            issues,
            harmonized_root,
            quarantine_root,
            provenance_extra={
                "target_rate_hz": harmonization_config.target_rate_hz,
                "filter_low_hz": harmonization_config.filter_low_hz,
                "filter_high_hz": harmonization_config.filter_high_hz,
                "filter_order": harmonization_config.filter_order,
            },
        )

        if issues:
            n_quarantined += 1
        else:
            n_written += 1

        duration_s = (
            len(harmonized_signal) / harmonization_config.target_rate_hz
            if len(harmonized_signal) > 0
            else 0.0
        )

        manifest_rows.append(ManifestRow(
            dataset=dataset,
            subject_id=trial.metadata.subject_id,
            activity_code=trial.metadata.activity_code,
            trial_id=trial.metadata.trial_id,
            label=trial.metadata.label,
            duration_s=duration_s,
            sample_rate_hz=harmonization_config.target_rate_hz,
            accepted=(len(issues) == 0),
            calibration_source=calibration.source,
            harmonized_path=str(written_path),
            fall_onset_frame=trial.metadata.fall_onset_frame,
            fall_impact_frame=trial.metadata.fall_impact_frame,
        ))

    if manifest_path is not None:
        write_manifest(manifest_rows, manifest_path)

    return HarmonizationSummary(
        n_trials_total=len(trials),
        n_written=n_written,
        n_quarantined=n_quarantined,
        calibration_source_counts=summarize_calibration_sources(resolved_calibrations),
    )
