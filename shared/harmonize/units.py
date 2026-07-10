"""Unit conversion layer.

Every dataset's raw signal arrives in whatever units its hardware
natively produces (KFall: already g / deg-per-s; SisFall: raw ADC
integers; FallAllD: vendor-specific, TBD). This module gives every
dataset a common `UnitConverter.convert()` interface so the rest of the
harmonization pipeline never has to special-case "which dataset is
this" -- it just calls `.convert()` and gets back a signal in g / deg-per-s.

KFall's converter is a verified no-op (not an assumed one -- see
test_units.py). SisFall's converter (Stage 5) does real ADC-to-physical
conversion using the scale factors from SisFall's own Readme.txt.
FallAllD's converter is a later-stage concern, not built here, but this
interface is what it will plug into without changing any calling code.
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


# SisFall's raw files are signed ADC integers from three sensors, not
# physical units. Per the dataset's own Readme.txt (confirmed against
# the real downloaded file, not just the paper): to convert raw bits
# (AD) to physical units, physical = [(2*Range)/(2^Resolution)] * AD.
# Same formula for both acceleration (-> g) and rotation (-> deg/s),
# just with each sensor's own Range/Resolution plugged in.
_SISFALL_ADXL345_SCALE = (2 * 16) / (2 ** 13)     # accelerometer #1: +-16g, 13-bit
_SISFALL_ITG3200_SCALE = (2 * 2000) / (2 ** 16)   # gyroscope:       +-2000 deg/s, 16-bit
_SISFALL_MMA8451Q_SCALE = (2 * 8) / (2 ** 14)     # accelerometer #2: +-8g, 14-bit


class SisFallUnitConverter:
    """Converts SisFall's raw ADC integers to physical units (g, deg/s).

    SisFall's device has TWO accelerometers (ADXL345, MMA8451Q) but the
    rest of this pipeline expects a single `acc_x/y/z` triplet, matching
    every other dataset. Which one becomes "the" accelerometer is a
    real decision, not a technicality -- resolved here by following the
    SisFall paper's own methodology: "Only acceleration data acquired
    with the ADXL345 sensor was used in this work, as it is energy
    efficient and provides the larger span" (Sucerquia et al., 2017).
    So ADXL345 -> acc_x/y/z (the channel the rest of the pipeline
    consumes), ITG3200 -> gyro_x/y/z, and MMA8451Q is converted too but
    kept under its own `mma_acc_*` columns -- preserved through this
    step for anyone who wants it later, then dropped during
    `pipeline.harmonize_trial`'s existing channel restriction to
    exactly `time_s` + acc_*/gyro_*, the same way KFall's Euler columns
    are archived-then-dropped rather than deleted here.
    """

    def convert(self, signal: pd.DataFrame) -> pd.DataFrame:
        out = signal.copy()

        out["acc_x"] = signal["raw_adxl_acc_x"] * _SISFALL_ADXL345_SCALE
        out["acc_y"] = signal["raw_adxl_acc_y"] * _SISFALL_ADXL345_SCALE
        out["acc_z"] = signal["raw_adxl_acc_z"] * _SISFALL_ADXL345_SCALE

        out["gyro_x"] = signal["raw_gyro_x"] * _SISFALL_ITG3200_SCALE
        out["gyro_y"] = signal["raw_gyro_y"] * _SISFALL_ITG3200_SCALE
        out["gyro_z"] = signal["raw_gyro_z"] * _SISFALL_ITG3200_SCALE

        out["mma_acc_x"] = signal["raw_mma_acc_x"] * _SISFALL_MMA8451Q_SCALE
        out["mma_acc_y"] = signal["raw_mma_acc_y"] * _SISFALL_MMA8451Q_SCALE
        out["mma_acc_z"] = signal["raw_mma_acc_z"] * _SISFALL_MMA8451Q_SCALE

        return out


_REGISTRY: dict[str, type[UnitConverter]] = {
    "kfall": KFallUnitConverter,
    "sisfall": SisFallUnitConverter,
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
