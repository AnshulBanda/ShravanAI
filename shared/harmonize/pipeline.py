"""Harmonization orchestrator.

Composes the four Stage 3 building blocks into one per-trial function,
in the frozen order: unit conversion -> resample -> axis alignment ->
filter.

IMPORTANT, non-obvious consequence of this order + the 0.5-20 Hz
band-pass filter: gravity is (by definition) a 0 Hz / DC signal. Axis
alignment correctly rotates it onto the canonical vertical axis, but
the subsequent band-pass filter then REMOVES that near-DC content --
same as it removes any slow postural drift, by design (see
PROJECT_CHECKPOINT.md). So a harmonized trial's output will NOT show a
persistent ~1g bias on acc_z, even for a perfectly still trial. This is
intentional: the filter's job is to isolate movement dynamics, not
preserve absolute orientation. Don't be alarmed if you check a
harmonized T01 trial and acc_z isn't sitting near 1.0 -- check that the
MOVEMENT-frequency content lines up on the right axis instead (see
test_pipeline.py for how that's verified).

Channel restriction: the output only contains time_s + the six
acc_*/gyro_* columns, even if the input trial had extra channels (e.g.
KFall's Euler angles) -- this is what makes the harmonized output
directly comparable across KFall/SisFall/FallAllD later. Callers who
want KFall's Euler angles for a KFall-only experiment should read them
from the original trial.signal directly, before harmonization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from shared.harmonize.axis_alignment import CalibrationResult, apply_rotation
from shared.harmonize.filtering import apply_bandpass_filter
from shared.harmonize.resample import resample_signal
from shared.harmonize.units import ACCEL_COLUMNS, GYRO_COLUMNS, get_unit_converter

_CANONICAL_COLUMNS = ["time_s"] + ACCEL_COLUMNS + GYRO_COLUMNS


@dataclass
class HarmonizationConfig:
    target_rate_hz: float = 100.0
    filter_low_hz: float = 0.5
    filter_high_hz: float = 20.0
    filter_order: int = 4


def harmonize_trial(
    trial: Any,
    calibration: CalibrationResult,
    config: HarmonizationConfig,
) -> pd.DataFrame:
    """Run one trial through the full harmonization pipeline.

    `trial` must have `.signal` (a DataFrame with time_s, acc_*, gyro_*
    columns) and `.metadata.dataset` (str, e.g. "kfall") and
    `.metadata.native_rate_hz` (float). Returns a new DataFrame with
    exactly `_CANONICAL_COLUMNS`, at `config.target_rate_hz`, aligned
    and filtered.
    """
    converter = get_unit_converter(trial.metadata.dataset)
    converted = converter.convert(trial.signal)

    resampled = resample_signal(
        converted,
        native_rate_hz=trial.metadata.native_rate_hz,
        target_rate_hz=config.target_rate_hz,
    )

    aligned = apply_rotation(resampled, calibration.rotation)

    filter_columns = ACCEL_COLUMNS + GYRO_COLUMNS
    filtered = apply_bandpass_filter(
        aligned,
        columns=filter_columns,
        sample_rate_hz=config.target_rate_hz,
        low_hz=config.filter_low_hz,
        high_hz=config.filter_high_hz,
        order=config.filter_order,
    )

    return filtered[_CANONICAL_COLUMNS].reset_index(drop=True)
