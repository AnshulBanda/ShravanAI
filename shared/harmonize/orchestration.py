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
from typing import Any, Optional

from shared.harmonize.axis_alignment import (
    STANDING_INITIATED_TASK_IDS,
    T01_TASK_ID,
    calibrate_subject,
    resolve_group_fallback,
    summarize_calibration_sources,
)
from shared.harmonize.pipeline import HarmonizationConfig, harmonize_trial
from shared.harmonize.units import get_unit_converter
from shared.harmonize.validation import validate_harmonized_trial
from shared.harmonize.writer import write_harmonized_trial
from shared.io.readers_kfall import load_all_trials as _load_kfall_trials
from shared.io.readers_sisfall import load_all_trials as _load_sisfall_trials
from shared.manifest import ManifestRow, write_manifest

# Registry of per-dataset trial loaders. Same pattern as units.py's
# converter registry -- add one entry here per dataset's reader.
_TRIAL_LOADERS = {
    "kfall": _load_kfall_trials,
    "sisfall": _load_sisfall_trials,
}

# SisFall activity codes that plausibly BEGIN from a standing posture
# before movement starts: D07-D10 (sit in a chair, which starts
# standing), D15-D17 (standing bend / standing-into-car activities).
# UNVERIFIED against real SisFall data as of this writing -- this is an
# assumption carried over from how KFall's STANDING_INITIATED_TASK_IDS
# was chosen, not yet confirmed the same way Task 3.4 confirmed KFall's
# via real stationarity checks. Needs a Task-3.11-style visual QA pass
# before fully trusting it -- see PROJECT_CHECKPOINT.md's Stage 5 section.
_SISFALL_STANDING_INITIATED_TASK_IDS = frozenset({7, 8, 9, 10, 15, 16, 17})

# Per-dataset calibration wiring: which task_id (if any) is this
# dataset's dedicated "stand still" trial, and which task_ids are
# reasonable auto-detect fallback candidates. SisFall has no dedicated
# calibration trial at all (see axis_alignment.py's
# primary_calibration_task_id docstring for why this matters -- SisFall's
# D01 happens to also be task_id 1, purely coincidentally, and must NOT
# be treated as a T01-equivalent).
_CALIBRATION_CONFIG = {
    "kfall": {
        "primary_calibration_task_id": T01_TASK_ID,
        "standing_initiated_task_ids": STANDING_INITIATED_TASK_IDS,
    },
    "sisfall": {
        "primary_calibration_task_id": None,
        "standing_initiated_task_ids": _SISFALL_STANDING_INITIATED_TASK_IDS,
    },
}


@dataclass
class HarmonizationSummary:
    n_trials_total: int
    n_written: int
    n_quarantined: int
    calibration_source_counts: dict[str, int]


@dataclass
class _CalibrationView:
    """Lightweight wrapper exposing a trial's UNIT-CONVERTED signal
    under `.signal`, alongside its original `.metadata`, for feeding
    into `calibrate_subject`.

    Why this exists: `axis_alignment.py` (correctly) expects canonical
    `acc_x/y/z`, `gyro_x/y/z` column names -- it has no idea which
    dataset it's looking at. KFall's raw reader output happens to
    already use those names (its converter is a verified no-op), so
    calling `calibrate_subject` directly on raw KFall trials worked by
    coincidence. SisFall's raw reader output uses `raw_adxl_acc_x` etc.
    -- calling calibrate_subject on THAT raw signal is a hard crash
    (`KeyError: 'acc_x'`), caught by test_orchestration.py's SisFall
    end-to-end tests during Stage 5 wiring. The fix: always run each
    trial through its dataset's unit converter before calibration, via
    this wrapper -- never pass a raw, not-yet-converted signal into
    calibrate_subject again, regardless of dataset.

    This does NOT change what `harmonize_trial` does later in the main
    loop -- that still runs the converter itself, from each trial's
    original raw `.signal`. Converting twice (once here, once inside
    harmonize_trial) is mildly redundant but harmless, since converters
    are pure functions with no side effects (see units.py's
    "does not mutate the input" tests) -- and far simpler than plumbing
    a pre-converted signal through the rest of the pipeline.
    """
    signal: Any
    metadata: Any


def _calibration_view(trial: Any, converter: Any) -> _CalibrationView:
    return _CalibrationView(signal=converter.convert(trial.signal), metadata=trial.metadata)


def get_trial_loader(dataset: str):
    """Look up the trial-loading function for a given dataset name.

    Public (mirrors `units.get_unit_converter`) so other code -- e.g.
    `notebooks/stage3_visual_qa.py`'s QA script -- can load a dataset's
    trials the same way `run_harmonization` does, without importing a
    specific reader module directly or duplicating the registry.
    """
    if dataset not in _TRIAL_LOADERS:
        raise ValueError(
            f"No trial loader registered for dataset {dataset!r}. "
            f"Known datasets: {sorted(_TRIAL_LOADERS)}"
        )
    return _TRIAL_LOADERS[dataset]


def resolve_calibrations(dataset: str, trials: list[Any]) -> dict[str, Any]:
    """Two-pass per-subject calibration for a dataset's trials: try each
    subject's own trials first (dataset-appropriate primary-trial check,
    then auto-detect fallback), then group-average fallback for any
    subject that didn't succeed on its own.

    Public and reused by BOTH `run_harmonization` and the visual-QA
    script -- this used to be duplicated inline in each, which is
    exactly how the Stage 5 "calibration ran on the raw signal" and
    "wrong sample rate" bugs stayed fixed in one copy but not the
    other. Only one copy of this logic exists now.

    Raises ValueError if `dataset` has no registered calibration config.
    """
    if dataset not in _CALIBRATION_CONFIG:
        raise ValueError(
            f"No calibration config registered for dataset {dataset!r}. "
            f"Known datasets: {sorted(_CALIBRATION_CONFIG)}"
        )
    calibration_config = _CALIBRATION_CONFIG[dataset]
    unit_converter = get_unit_converter(dataset)

    by_subject: dict[str, list] = {}
    for trial in trials:
        by_subject.setdefault(trial.metadata.subject_id, []).append(trial)

    per_subject_calibration = {
        subject_id: calibrate_subject(
            [_calibration_view(t, unit_converter) for t in subject_trials],
            standing_initiated_task_ids=calibration_config["standing_initiated_task_ids"],
            # Calibration runs on the unit-converted (but NOT yet
            # resampled) signal, so the stationarity window must be
            # sized using each dataset's own NATIVE rate, not the
            # harmonization target rate -- using target_rate_hz here
            # would silently miscompute window sizes for any dataset
            # whose native rate differs from it (e.g. SisFall: 200 Hz
            # native vs. 100 Hz target). All of a subject's trials
            # share one dataset, so any trial's native_rate_hz is
            # representative.
            sample_rate_hz=subject_trials[0].metadata.native_rate_hz,
            primary_calibration_task_id=calibration_config["primary_calibration_task_id"],
        )
        for subject_id, subject_trials in by_subject.items()
    }
    return resolve_group_fallback(per_subject_calibration)


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

    Raises ValueError if `dataset` has no registered trial loader or
    calibration config.
    """
    trials = get_trial_loader(dataset)(sensor_root, label_root)
    resolved_calibrations = resolve_calibrations(dataset, trials)

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
