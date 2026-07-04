"""Reader for the KFall dataset's native format.

Parses raw per-trial sensor CSVs and per-subject label Excel files into
a common in-memory representation used by the harmonization pipeline.

Scope note: this module does NOT do unit conversion, resampling,
filtering, or channel restriction -- it only parses the native KFall
format faithfully (including the Euler-angle channels, which later get
dropped or archived during harmonization, not here). Keeping the reader
this "dumb" is deliberate: it's the one piece of code that should never
need to change once it's verified against real files, regardless of how
the harmonization design evolves.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# KFall sensor CSVs are sampled at a fixed native rate.
KFALL_NATIVE_RATE_HZ = 100.0

# Task IDs T22-T36 are simulated falls (per the KFall protocol); every
# other task ID is an activity of daily living.
FALL_TASK_IDS = frozenset(range(22, 37))

TRIAL_FILENAME_RE = re.compile(r"^(SA\d+)T(\d+)R(\d+)\.csv$", re.IGNORECASE)

EXPECTED_SENSOR_COLUMNS = [
    "TimeStamp", "FrameCounter",
    "AccX", "AccY", "AccZ",
    "GyroX", "GyroY", "GyroZ",
    "EulerX", "EulerY", "EulerZ",
]

_RENAME_MAP = {
    "AccX": "acc_x", "AccY": "acc_y", "AccZ": "acc_z",
    "GyroX": "gyro_x", "GyroY": "gyro_y", "GyroZ": "gyro_z",
    "EulerX": "euler_x", "EulerY": "euler_y", "EulerZ": "euler_z",
}

_SIGNAL_COLUMN_ORDER = [
    "time_s",
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "euler_x", "euler_y", "euler_z",
]


@dataclass
class TrialMetadata:
    dataset: str
    subject_id: str          # e.g. "SA06"
    activity_code: str       # e.g. "T20"
    trial_id: str            # e.g. "R01"
    task_id: int             # e.g. 20
    label: str                # "fall" or "adl"
    native_rate_hz: float
    source_path: str
    fall_onset_frame: Optional[int] = None
    fall_impact_frame: Optional[int] = None


@dataclass
class ParsedTrial:
    signal: pd.DataFrame     # columns: time_s, acc_x..gyro_z, euler_x..euler_z
    metadata: TrialMetadata


def parse_trial_filename(filename: str) -> tuple[str, int, str]:
    """Extract (subject_id, task_id, trial_id) from a KFall sensor filename.

    e.g. "SA06T20R01.csv" -> ("SA06", 20, "R01")
    """
    match = TRIAL_FILENAME_RE.match(filename)
    if not match:
        raise ValueError(
            f"Filename does not match expected KFall pattern 'SAxxTyyRzz.csv': {filename!r}"
        )
    subject_id, task_str, trial_str = match.groups()
    return subject_id.upper(), int(task_str), f"R{trial_str}"


def read_sensor_csv(path: Path) -> pd.DataFrame:
    """Parse one KFall sensor CSV into a standardized DataFrame.

    Units are left exactly as KFall provides them (g, deg/s, deg) --
    unit conversion is a harmonization-stage concern, not a reader concern.
    """
    df = pd.read_csv(path)

    missing = set(EXPECTED_SENSOR_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing expected columns {sorted(missing)}")

    df = df.rename(columns=_RENAME_MAP)
    df["time_s"] = df["TimeStamp"].astype(float)

    return df[_SIGNAL_COLUMN_ORDER].reset_index(drop=True)


def read_label_file(path: Path) -> pd.DataFrame:
    """Parse a KFall per-subject label Excel file.

    Column names are normalized (lowercased, spaces -> underscores)
    rather than assumed verbatim, since the exact header text hasn't
    been confirmed against a real KFall label file yet in this repo --
    verify column names the first time this runs against real data, and
    tighten `_find_label_columns` below if the heuristic matching in
    there picks the wrong column.
    """
    raw = pd.read_excel(path)
    raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]
    return raw


def _find_label_columns(label_df: pd.DataFrame) -> Optional[tuple[str, str, str, str]]:
    """Heuristically locate (task_col, trial_col, onset_col, impact_col) by
    substring match on normalized column names. Returns None if any are
    missing, so callers can fail soft rather than throw on unexpected
    ADL-only label sheets.
    """
    task_cols = [c for c in label_df.columns if "task" in c]
    trial_cols = [c for c in label_df.columns if "trial" in c]
    onset_cols = [c for c in label_df.columns if "onset" in c]
    impact_cols = [c for c in label_df.columns if "impact" in c]

    if not (task_cols and trial_cols and onset_cols and impact_cols):
        return None
    return task_cols[0], trial_cols[0], onset_cols[0], impact_cols[0]


def _label_lookup(
    label_df: pd.DataFrame, task_id: int, trial_id: str
) -> tuple[Optional[int], Optional[int]]:
    """Find the onset/impact frame for a given task/trial in a parsed label sheet.

    Returns (None, None) if no matching row is found, which is the
    expected/correct outcome for ADL trials that have no onset/impact
    annotation at all.
    """
    columns = _find_label_columns(label_df)
    if columns is None:
        return None, None
    task_col, trial_col, onset_col, impact_col = columns

    task_str = f"T{task_id:02d}"
    matches = label_df[
        label_df[task_col].astype(str).str.upper().str.strip().isin([task_str, str(task_id)])
        & label_df[trial_col].astype(str).str.upper().str.strip().str.contains(trial_id, na=False)
    ]
    if matches.empty:
        return None, None

    row = matches.iloc[0]
    onset = int(row[onset_col]) if pd.notna(row[onset_col]) else None
    impact = int(row[impact_col]) if pd.notna(row[impact_col]) else None
    return onset, impact


def load_trial(sensor_csv_path: Path, label_df: Optional[pd.DataFrame]) -> ParsedTrial:
    """Load one full trial: signal + metadata, cross-referencing the label
    sheet if one was provided for this trial's subject.
    """
    subject_id, task_id, trial_id = parse_trial_filename(sensor_csv_path.name)
    signal = read_sensor_csv(sensor_csv_path)

    onset, impact = (None, None)
    if label_df is not None:
        onset, impact = _label_lookup(label_df, task_id, trial_id)

    label = "fall" if task_id in FALL_TASK_IDS else "adl"

    metadata = TrialMetadata(
        dataset="kfall",
        subject_id=subject_id,
        activity_code=f"T{task_id:02d}",
        trial_id=trial_id,
        task_id=task_id,
        label=label,
        native_rate_hz=KFALL_NATIVE_RATE_HZ,
        source_path=str(sensor_csv_path),
        fall_onset_frame=onset,
        fall_impact_frame=impact,
    )
    return ParsedTrial(signal=signal, metadata=metadata)


def discover_trials(sensor_root: Path) -> list[Path]:
    """List all sensor CSV files under a KFall sensor_data root, sorted
    for reproducible iteration order.
    """
    return sorted(Path(sensor_root).glob("SA*/SA*T*R*.csv"))


def load_all_trials(sensor_root: Path, label_root: Path) -> list[ParsedTrial]:
    """Load every trial found under sensor_root, matched against its
    subject's per-subject label file (if one exists under label_root).
    """
    sensor_root, label_root = Path(sensor_root), Path(label_root)
    trials: list[ParsedTrial] = []
    label_cache: dict[str, Optional[pd.DataFrame]] = {}

    for csv_path in discover_trials(sensor_root):
        subject_id, _, _ = parse_trial_filename(csv_path.name)
        if subject_id not in label_cache:
            label_path = label_root / f"{subject_id}_label.xlsx"
            label_cache[subject_id] = read_label_file(label_path) if label_path.exists() else None
        trials.append(load_trial(csv_path, label_cache[subject_id]))

    return trials
