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


# NOTE on naming variants: the official KFall release names sensor CSVs
# "SAxxTyyRzz.csv" (e.g. SA06T01R01.csv). At least one Kaggle mirror
# (usmanabbasi2002/kfall-dataset) drops the "A" from the *filename* while
# keeping it in the parent folder name -- i.e. folder "SA06/" contains
# "S06T01R01.csv", not "SA06T01R01.csv". Label files on that same mirror
# keep the full "SAxx_label.xlsx" naming, so only sensor CSV filenames
# are affected. The "A" is made optional here and the subject number is
# always re-normalized to canonical "SAxx" form on output, so callers
# never need to know which naming variant a given mirror used.
TRIAL_FILENAME_RE = re.compile(r"^S(A)?(\d+)T(\d+)R(\d+)\.csv$", re.IGNORECASE)

EXPECTED_SENSOR_COLUMNS = [
    "TimeStamp(s)", "FrameCounter",
    "AccX", "AccY", "AccZ",
    "GyrX", "GyrY", "GyrZ",
    "EulerX", "EulerY", "EulerZ",
]

_RENAME_MAP = {
    "AccX": "acc_x", "AccY": "acc_y", "AccZ": "acc_z",
    "GyrX": "gyro_x", "GyrY": "gyro_y", "GyrZ": "gyro_z",
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

    Accepts both the official "SAxxTyyRzz.csv" naming and the
    "SxxTyyRzz.csv" (dropped-"A") variant seen on at least one Kaggle
    mirror -- either way, the returned subject_id is always normalized
    to canonical "SAxx" form.

    e.g. "SA06T20R01.csv" -> ("SA06", 20, "R01")
         "S06T20R01.csv"  -> ("SA06", 20, "R01")
    """
    match = TRIAL_FILENAME_RE.match(filename)
    if not match:
        raise ValueError(
            f"Filename does not match expected KFall pattern "
            f"'SAxxTyyRzz.csv' (or the dropped-A 'SxxTyyRzz.csv' variant): {filename!r}"
        )
    _has_a, subject_num, task_str, trial_str = match.groups()
    subject_id = f"SA{int(subject_num):02d}"
    return subject_id, int(task_str), f"R{trial_str}"


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
    df["time_s"] = df["TimeStamp(s)"].astype(float)

    return df[_SIGNAL_COLUMN_ORDER].reset_index(drop=True)


_TASK_CODE_T_RE = re.compile(r"^T(\d+)$", re.IGNORECASE)
_TASK_CODE_F_RE = re.compile(r"^F(\d+)\s*\((\d+)\)$", re.IGNORECASE)


def _resolve_official_task_id(raw_code) -> int:
    """Convert a label sheet's task-code cell into the canonical KFall
    task ID (an int, e.g. 22 for T22).

    Real KFall label spreadsheets encode fall trials as "F01 (20)" --
    neither the F-number nor the parenthetical number is the canonical
    task ID used in sensor filenames (T22-T36). Cross-checking all 15
    fall-type descriptions on a real SA06 label file against the
    official KFall task descriptions confirmed: canonical task ID =
    parenthetical_number + 2, holding the invariant
    F_number + 19 == parenthetical_number across every row (verified,
    not assumed). This also accepts a plain "T22" form directly, in
    case some label file already uses canonical codes. Raises
    ValueError on anything else, or if the invariant is violated --
    both mean this mapping can't be trusted for whatever produced it,
    which is worse to silently get wrong than to fail loudly on.
    """
    text = str(raw_code).strip()

    t_match = _TASK_CODE_T_RE.match(text)
    if t_match:
        return int(t_match.group(1))

    f_match = _TASK_CODE_F_RE.match(text)
    if f_match:
        f_number, paren_number = int(f_match.group(1)), int(f_match.group(2))
        if f_number + 19 != paren_number:
            raise ValueError(
                f"Task code {raw_code!r} breaks the expected F-number/parenthetical "
                f"relationship (expected parenthetical == F_number + 19) -- the "
                f"canonical-task-ID mapping may not hold for this file; investigate "
                f"before trusting it."
            )
        return paren_number + 2

    raise ValueError(f"Unrecognized task code format in label sheet: {raw_code!r}")


def read_label_file(path: Path) -> pd.DataFrame:
    """Parse a KFall per-subject label Excel file.

    Column names are normalized (lowercased, spaces -> underscores).
    Real KFall label sheets merge the task-code and description cells
    across each task's repeated trial rows (Excel merged cells become
    blank/NaN on every row after the first when read via pandas) --
    those two columns are forward-filled here so every row is usable
    independently. A `resolved_task_id` column is added, mapping each
    row's raw task code (whichever format it's in) to the canonical
    KFall task ID -- see `_resolve_official_task_id`.
    """
    raw = pd.read_excel(path)
    raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]

    task_cols = [c for c in raw.columns if "task" in c]
    if not task_cols:
        return raw

    task_col = task_cols[0]
    raw[task_col] = raw[task_col].ffill()

    desc_cols = [c for c in raw.columns if "description" in c]
    if desc_cols:
        raw[desc_cols[0]] = raw[desc_cols[0]].ffill()

    resolved: list = []
    errors: list = []
    for idx, code in raw[task_col].items():
        try:
            resolved.append(_resolve_official_task_id(code))
        except ValueError as exc:
            errors.append(f"row {idx}: {exc}")
            resolved.append(None)

    if errors:
        raise ValueError(
            "Failed to resolve task code(s) in label file "
            f"{path.name}:\n" + "\n".join(errors)
        )

    raw["resolved_task_id"] = resolved
    return raw


def _label_lookup(
    label_df: pd.DataFrame, task_id: int, trial_id: str
) -> tuple[Optional[int], Optional[int]]:
    """Find the onset/impact frame for a given task/trial in a parsed label sheet.

    Returns (None, None) if no matching row is found, which is the
    expected/correct outcome for ADL trials that have no onset/impact
    annotation at all.
    """
    if "resolved_task_id" not in label_df.columns:
        return None, None

    trial_cols = [c for c in label_df.columns if "trial" in c]
    onset_cols = [c for c in label_df.columns if "onset" in c]
    impact_cols = [c for c in label_df.columns if "impact" in c]
    if not (trial_cols and onset_cols and impact_cols):
        return None, None
    trial_col, onset_col, impact_col = trial_cols[0], onset_cols[0], impact_cols[0]

    # trial_id arrives here as "R01" (parsed from the sensor filename);
    # real label sheets store it as a plain integer repetition number.
    trial_num_match = re.match(r"R?0*(\d+)", trial_id, re.IGNORECASE)
    if not trial_num_match:
        return None, None
    trial_num = int(trial_num_match.group(1))

    matches = label_df[
        (label_df["resolved_task_id"] == task_id)
        & (pd.to_numeric(label_df[trial_col], errors="coerce") == trial_num)
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

    Matches trial files by content (via TRIAL_FILENAME_RE) rather than a
    glob pattern on the filename prefix, since at least one real-world
    mirror uses a different filename convention than its own subject
    folder names (see the naming-variant note above TRIAL_FILENAME_RE).
    Relying on the regex here means any future naming quirk only needs
    a fix in one place.
    """
    sensor_root = Path(sensor_root)
    candidates = sensor_root.glob("SA*/*.csv")
    return sorted(p for p in candidates if TRIAL_FILENAME_RE.match(p.name))


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
