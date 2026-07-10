"""Reader for the SisFall dataset's native format.

Parses raw per-trial `.txt` files into the same common in-memory
representation `readers_kfall.py` uses (a `signal` DataFrame + a
`TrialMetadata` object), so the rest of the harmonization pipeline
doesn't need to know which dataset a trial came from.

Verified against real downloaded SisFall files (SisFall_dataset, per
the dataset's own Readme.txt) -- not assumed from the paper alone:
- 9 raw ADC columns per row, comma-separated, each line ending in `;`,
  no header row: accX,accY,accZ (ADXL345), gyroX,gyroY,gyroZ (ITG3200),
  accX,accY,accZ (MMA8451Q).
- Filenames: `<D##_or_F##>_<SUBJECT_ID>_<TRIAL_NO>.txt`, one folder per
  subject. `D` prefix = ADL, `F` prefix = fall. Confirmed against a
  real file listing: 38 subject folders present (SA01-SA23, SE01-SE15),
  matching the Readme's stated roster.
- Native rate: 200 Hz (KFall was already 100 Hz -- this is the first
  dataset where resampling in the harmonization pipeline does real work).
- Column count verified consistent (9) across a real sample of files.

Scope note, same as the KFall reader: this module does NOT do unit
conversion (raw ADC -> g/deg-per-s), resampling, or filtering -- it only
parses the native SisFall format faithfully. Raw ADC values are left as
signed integers in the returned signal; `shared/harmonize/units.py`'s
SisFallUnitConverter is what turns them into physical units.

Known, deliberate gap vs. KFall: SisFall has NO frame-level fall
onset/impact labels anywhere (confirmed against the real Readme.txt --
there is no separate label file, and the raw trial files carry no
annotation columns). Every SisFall trial's `fall_onset_frame` and
`fall_impact_frame` are therefore always None, including fall trials --
this is expected, not a parsing failure, and is exactly why the
blueprint restricts the prediction pipeline (which needs onset/impact
frames) to KFall only. `shared/manifest.py`'s `query_prediction_trials`
already filters on `dataset == "kfall"`, so this requires no special
handling downstream.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

SISFALL_NATIVE_RATE_HZ = 200.0

# Raw column order exactly as SisFall's Readme.txt documents it. Kept
# with a "raw_" prefix (rather than acc_x/gyro_x etc.) because these are
# NOT yet in physical units -- using the same column names as the
# harmonized/physical-unit signal would risk something downstream
# accidentally treating raw ADC counts as g's before conversion.
RAW_COLUMN_ORDER = [
    "raw_adxl_acc_x", "raw_adxl_acc_y", "raw_adxl_acc_z",
    "raw_gyro_x", "raw_gyro_y", "raw_gyro_z",
    "raw_mma_acc_x", "raw_mma_acc_y", "raw_mma_acc_z",
]

TRIAL_FILENAME_RE = re.compile(
    r"^([DF])(\d+)_([A-Z]{2}\d+)_R(\d+)\.txt$", re.IGNORECASE
)


@dataclass
class TrialMetadata:
    dataset: str
    subject_id: str          # e.g. "SA01", "SE06"
    activity_code: str       # e.g. "D01", "F05"
    trial_id: str            # e.g. "R01"
    task_id: int             # e.g. 1 -- see module docstring's warning
                              # in the harmonization-wiring discussion:
                              # ADL and fall codes both start at 1
                              # (D01 vs F01), so this is NOT
                              # cross-dataset-comparable to KFall's
                              # task_id and must not be used to infer
                              # "this is a T01-equivalent calibration
                              # trial" the way KFall's task_id is.
    label: str                # "fall" or "adl"
    native_rate_hz: float
    source_path: str
    fall_onset_frame: Optional[int] = None   # always None for SisFall
    fall_impact_frame: Optional[int] = None  # always None for SisFall


@dataclass
class ParsedTrial:
    signal: pd.DataFrame     # columns: time_s, raw_adxl_acc_*, raw_gyro_*, raw_mma_acc_*
    metadata: TrialMetadata


def parse_trial_filename(filename: str) -> tuple[str, str, int, str]:
    """Extract (activity_code, subject_id, task_id, trial_id) from a
    SisFall sensor filename.

    e.g. "D01_SA01_R01.txt" -> ("D01", "SA01", 1, "R01")
         "F05_SE06_R04.txt" -> ("F05", "SE06", 5, "R04")
    """
    match = TRIAL_FILENAME_RE.match(filename)
    if not match:
        raise ValueError(
            f"Filename does not match expected SisFall pattern "
            f"'<D|F><NN>_<SUBJECT>_R<NN>.txt': {filename!r}"
        )
    prefix, code_num, subject_id, trial_num = match.groups()
    activity_code = f"{prefix.upper()}{int(code_num):02d}"
    return activity_code, subject_id.upper(), int(code_num), f"R{trial_num}"


def read_sensor_txt(path: Path) -> pd.DataFrame:
    """Parse one SisFall raw `.txt` file into a standardized DataFrame.

    Values are left as raw signed ADC integers -- unit conversion is a
    harmonization-stage concern (SisFallUnitConverter), not a reader
    concern, same separation of responsibility as the KFall reader.

    Each line is `int,int,int,int,int,int,int,int,int;` (9
    comma-separated values, trailing semicolon, no header). Lines are
    stripped of the trailing `;` and surrounding whitespace before
    splitting -- values themselves may have leading spaces for
    right-justified alignment in the raw file, so int() (not a
    fixed-width slice) is used to parse each field.
    """
    rows: list[list[int]] = []
    with open(path) as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue  # tolerate a trailing blank line at EOF
            line = line.rstrip(";")
            fields = [v.strip() for v in line.split(",")]
            if len(fields) != len(RAW_COLUMN_ORDER):
                raise ValueError(
                    f"{path.name}, line {line_num}: expected "
                    f"{len(RAW_COLUMN_ORDER)} columns, got {len(fields)}: {raw_line!r}"
                )
            try:
                rows.append([int(v) for v in fields])
            except ValueError as exc:
                raise ValueError(
                    f"{path.name}, line {line_num}: non-integer value in {raw_line!r}"
                ) from exc

    df = pd.DataFrame(rows, columns=RAW_COLUMN_ORDER)
    df["time_s"] = df.index / SISFALL_NATIVE_RATE_HZ
    return df[["time_s"] + RAW_COLUMN_ORDER]


def load_trial(sensor_txt_path: Path) -> ParsedTrial:
    """Load one full SisFall trial: signal + metadata.

    No label file cross-referencing (unlike KFall) -- there isn't one
    for SisFall. Fall/ADL comes straight from the filename prefix.
    """
    activity_code, subject_id, task_id, trial_id = parse_trial_filename(
        sensor_txt_path.name
    )
    signal = read_sensor_txt(sensor_txt_path)
    label = "fall" if activity_code.startswith("F") else "adl"

    metadata = TrialMetadata(
        dataset="sisfall",
        subject_id=subject_id,
        activity_code=activity_code,
        trial_id=trial_id,
        task_id=task_id,
        label=label,
        native_rate_hz=SISFALL_NATIVE_RATE_HZ,
        source_path=str(sensor_txt_path),
    )
    return ParsedTrial(signal=signal, metadata=metadata)


def discover_trials(sensor_root: Path) -> list[Path]:
    """List all trial `.txt` files under a SisFall dataset root
    (`SisFall_dataset/`), sorted for reproducible iteration order.

    Matches by content (via TRIAL_FILENAME_RE), not a glob on the
    filename prefix -- excludes `Readme.txt` and anything else that
    doesn't match the real trial naming convention, same defensive
    approach as the KFall reader's `discover_trials`.
    """
    sensor_root = Path(sensor_root)
    candidates = sensor_root.glob("*/*.txt")
    return sorted(p for p in candidates if TRIAL_FILENAME_RE.match(p.name))


def load_all_trials(sensor_root: Path, label_root: Optional[Path] = None) -> list[ParsedTrial]:
    """Load every trial found under sensor_root.

    `label_root` is accepted (and ignored) purely so this function has
    the same signature as `readers_kfall.load_all_trials` and can be
    registered in `orchestration._TRIAL_LOADERS` without a wrapper.
    """
    sensor_root = Path(sensor_root)
    return [load_trial(p) for p in discover_trials(sensor_root)]
