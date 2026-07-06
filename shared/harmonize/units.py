"""Unit conversion layer.

Every dataset's raw signal arrives in whatever units its hardware
natively produces (KFall: already g / deg-per-s; SisFall: raw ADC
integers; FallAllD: vendor-specific, TBD). This module gives every
dataset a common `UnitConverter.convert()` interface so the rest of the
harmonization pipeline never has to special-case "which dataset is
this" -- it just calls `.convert()` and gets back a signal in g / deg-per-s.

KFall's converter is a verified no-op (not an assumed one -- see
test_units.py). SisFall's real ADC->physical-unit converter is a Stage 5
concern, not built here, but this interface is what it will plug into
without changing any calling code.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd

# Channels this pipeline cares about converting. Euler angle channels
# (present only in KFall) are intentionally left untouched here --
# they're a KFall-only concern the harmonization pipeline restricts out
# entirely in a later step (see blueprint sec 3, "channel restriction"),
# not something unit conversion needs to know about.
ACCEL_COLUMNS = ["acc_x", "acc_y", "acc_z"]
GYRO_COLUMNS = ["gyro_x", "gyro_y", "gyro_z"]


class UnitConverter(Protocol):
    """Common interface every per-dataset unit converter implements."""

    def convert(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of `signal` with acc_*/gyro_* columns in
        physical units (g, deg/s). Must not mutate the input in place.
        """
        ...


class KFallUnitConverter:
    """KFall's sensor CSVs are already pre-fused into physical units
    (g for acceleration, deg/s for gyroscope) by the LPMS-B2's onboard
    processing -- so this converter is a genuine no-op. It still exists
    as a real class (rather than being skipped) so every dataset goes
    through the same `get_unit_converter(...).convert(...)` call site.
    """

    def convert(self, signal: pd.DataFrame) -> pd.DataFrame:
        return signal.copy()


_REGISTRY: dict[str, type[UnitConverter]] = {
    "kfall": KFallUnitConverter,
}


def get_unit_converter(dataset: str) -> UnitConverter:
    """Look up the unit converter for a given dataset name.

    Raises ValueError with the list of known datasets if `dataset` isn't
    registered yet -- e.g. before SisFall's converter is built in Stage 5.
    """
    key = dataset.lower()
    if key not in _REGISTRY:
        known = sorted(_REGISTRY)
        raise ValueError(
            f"No unit converter registered for dataset {dataset!r}. "
            f"Known datasets: {known}"
        )
    return _REGISTRY[key]()
