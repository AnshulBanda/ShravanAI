"""Detection pipeline: dataset construction (windowing + labeling).

Turns the trial-level manifest (shared/manifest.py, already covering
both KFall and SisFall harmonization) into a WINDOW-level manifest --
one row per fixed-length window, with its label and enough provenance
to load the actual signal on demand. Kept as a lightweight metadata
index, same philosophy as the trial manifest: `build_windows_manifest`
never loads a harmonized parquet's actual signal data, only its
`duration_s`/`sample_rate_hz` metadata already in the trial manifest.
Loading real signal data happens window-by-window, on demand, via
`load_window` -- so this stays cheap to build/rebuild even across
thousands of trials, and doesn't materialize a second (much larger)
copy of the signal data on disk.

Real bug this design avoids: KFall and SisFall subject IDs COLLIDE --
both datasets use "SA01".."SA23"/"SE01".."SE15"-style IDs, so KFall's
SA06 and SisFall's SA06 are DIFFERENT people who happen to share a
subject_id string. Any subject-level grouping (LOSO/LODO splits,
per-subject anything) that groups by raw `subject_id` alone would
silently conflate them -- a real subject-leakage bug, not a
hypothetical one, given this project's own two datasets literally use
overlapping ID ranges. Every window record carries a `global_subject_id`
(f"{dataset}_{subject_id}") specifically so downstream split code has
no reason to ever group by bare `subject_id` again.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from detection.windowing import WindowingConfig, generate_window_specs
from shared.harmonize.units import ACCEL_COLUMNS, GYRO_COLUMNS
from shared.manifest import query_detection_trials

# Canonical per-window channel order. Every harmonized trial (both
# datasets) already has exactly these 6 columns plus time_s -- see
# pipeline.py's channel restriction -- so no dataset-specific handling
# is needed here.
CHANNELS = ACCEL_COLUMNS + GYRO_COLUMNS


@dataclass
class WindowRecord:
    dataset: str
    subject_id: str
    global_subject_id: str
    activity_code: str
    trial_id: str
    label: int              # 0 = adl, 1 = fall -- inherited from the WHOLE source trial (coarse; see module docstring's label-noise note)
    window_index: int       # 0-based index of this window within its source trial
    start_frame: int
    end_frame: int           # exclusive, BEFORE padding
    n_real_samples: int
    n_pad_samples: int
    harmonized_path: str


def build_windows_manifest(
    trial_manifest_df: pd.DataFrame,
    config: Optional[WindowingConfig] = None,
    datasets: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Build the window-level manifest for the detection pipeline.

    `trial_manifest_df` is the trial-level manifest from
    `shared.manifest.load_manifest` -- this function filters it via
    `query_detection_trials` (both datasets, accepted trials only) and
    expands each surviving trial into its window boundaries.

    Label is inherited whole-trial -> every window (§4 of the
    blueprint's Pipeline 1 spec): this is a known, documented coarse
    label, not something this function tries to refine. A fall-trial
    window covering only pre-fall walking gets label=1 same as one
    covering the actual impact -- expect resulting label noise near
    trial boundaries, exactly as the blueprint flags.
    """
    config = config or WindowingConfig()
    detection_trials = query_detection_trials(trial_manifest_df, datasets=datasets)

    records: list[WindowRecord] = []
    for _, row in detection_trials.iterrows():
        # Reconstructs the harmonized trial's exact sample count without
        # opening the file -- valid because duration_s was itself
        # computed as `len(signal) / target_rate_hz` when the trial
        # manifest was written (orchestration.py), so multiplying back
        # recovers the same integer (up to float rounding, handled by
        # round()).
        n_samples = round(row["duration_s"] * row["sample_rate_hz"])
        window_specs = generate_window_specs(n_samples, config)

        label = 1 if row["label"] == "fall" else 0
        global_subject_id = f'{row["dataset"]}_{row["subject_id"]}'

        for window_index, spec in enumerate(window_specs):
            records.append(WindowRecord(
                dataset=row["dataset"],
                subject_id=row["subject_id"],
                global_subject_id=global_subject_id,
                activity_code=row["activity_code"],
                trial_id=row["trial_id"],
                label=label,
                window_index=window_index,
                start_frame=spec.start_frame,
                end_frame=spec.end_frame,
                n_real_samples=spec.n_real_samples,
                n_pad_samples=spec.n_pad_samples,
                harmonized_path=row["harmonized_path"],
            ))

    if not records:
        # Explicit empty-frame with the right columns, rather than
        # letting pd.DataFrame([]) produce a columnless frame that
        # would break any code expecting these column names to exist.
        return pd.DataFrame([asdict(r) for r in [_EMPTY_RECORD]]).iloc[0:0]

    return pd.DataFrame([asdict(r) for r in records])


_EMPTY_RECORD = WindowRecord(
    dataset="", subject_id="", global_subject_id="", activity_code="", trial_id="",
    label=0, window_index=0, start_frame=0, end_frame=0,
    n_real_samples=0, n_pad_samples=0, harmonized_path="",
)


def load_window(
    window_row: pd.Series,
    window_length_samples: int,
    signal_cache: Optional[dict[str, pd.DataFrame]] = None,
) -> np.ndarray:
    """Load one window's actual signal data as a
    (window_length_samples, 6) array in CHANNELS order.

    `signal_cache`: an optional dict the CALLER owns and reuses across
    many `load_window` calls (e.g. one per training epoch's worth of
    windows) to avoid re-reading the same harmonized parquet file
    once per window -- a single trial can produce dozens of
    overlapping windows (50% stride), so without this a naive loop
    would re-read one file dozens of times. Pass the same dict back in
    on every call within one pass over the windows manifest; a fresh
    empty dict starts a fresh cache (e.g. per epoch, if memory is a
    concern and you'd rather not hold every trial's signal in RAM at
    once across an entire dataset pass).

    Padding strategy for short/trailing windows (n_pad_samples > 0 per
    the windows manifest): EDGE-padding (repeat the last real sample),
    not zero-padding. Zero-padding would inject a sharp, artificial
    discontinuity into accelerometer data (a sudden drop to 0g, which
    looks like freefall to a model that's never seen real freefall
    data at that exact shape) -- edge-padding instead implies "the
    signal stayed at its last real value," which is a far more
    physically plausible assumption for a trial that just ended
    (subject settled into a final resting position) than a fabricated
    zero.
    """
    signal_cache = signal_cache if signal_cache is not None else {}
    path = window_row["harmonized_path"]

    if path not in signal_cache:
        signal_cache[path] = pd.read_parquet(path, columns=CHANNELS)
    signal_df = signal_cache[path]

    start, end = int(window_row["start_frame"]), int(window_row["end_frame"])
    real_segment = signal_df.iloc[start:end][CHANNELS].to_numpy(dtype=np.float32)

    n_pad = window_length_samples - len(real_segment)
    if n_pad <= 0:
        return real_segment[:window_length_samples]

    if len(real_segment) == 0:
        # Defensive: a window with zero real samples has nothing to
        # edge-pad from. Shouldn't occur given generate_window_specs
        # only ever produces n_real_samples >= 1 for trial_n_samples > 0,
        # but fail loudly rather than silently returning garbage if it does.
        raise ValueError(
            f"Window has 0 real samples (path={path}, start={start}, end={end}) "
            "-- cannot edge-pad from nothing."
        )

    pad_block = np.repeat(real_segment[-1:], n_pad, axis=0)
    return np.concatenate([real_segment, pad_block], axis=0)
