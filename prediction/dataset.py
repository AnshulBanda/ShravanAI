"""Prediction pipeline: dataset construction (dense windowing + onset/impact labeling).

Mirrors `detection/dataset.py`'s structure (window-level manifest,
loaded on demand rather than materialized) but differs in three ways
that matter enough to keep this a separate module rather than a
parametrized shared one, per the blueprint's explicit call-out
(§"what's genuinely separate"):
  1. Source trials: `shared.manifest.query_prediction_trials` (KFall
     only, onset/impact-eligible trials only) instead of
     `query_detection_trials` (all datasets).
  2. Windowing: `prediction.windowing.PredictionWindowingConfig`
     (dense, 1.0s/0.1s) instead of detection's 2.0s/1.0s.
  3. Labeling: `prediction.labelers.onset_impact_label` (3-class, per
     window, frame-precise) instead of detection's whole-trial binary
     label.

KNOWN GAP vs. the blueprint's aspirational spec, flagged rather than
silently worked around: blueprint Pipeline 2 §1 says prediction should
"keep KFall's full channel set (accel, gyro, and the pre-fused Euler
angles) -- no need to restrict channels ... since you're single-
dataset." In the ACTUAL harmonization pipeline as already built
(`shared/harmonize/pipeline.py`), Euler angles are dropped for EVERY
trial at harmonization time, before either pipeline sees the data --
see that module's own docstring: "Callers who want KFall's Euler
angles for a KFall-only experiment should read them from the original
trial.signal directly, before harmonization." So this module, like
detection, currently only has access to the 6 harmonized acc_*/gyro_*
channels (`CHANNELS` below) -- not Euler. Re-deriving Euler would mean
either re-harmonizing KFall with a per-dataset channel policy (a real
change to the already real-data-verified harmonization pipeline, not
a prediction-side change) or reading raw KFall files a second time,
bypassing the harmonized layer for this one pipeline only. Deliberately
NOT decided here -- worth a real conversation before touching
harmonization -- so for now the prediction pipeline proceeds on the
same 6-channel signal as detection, and this gap is tracked as an open
item rather than quietly built around.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd

from prediction.labelers import LABEL_TO_INT, onset_impact_label
from prediction.windowing import PredictionWindowingConfig, generate_window_specs
from shared.harmonize.units import ACCEL_COLUMNS, GYRO_COLUMNS
from shared.manifest import query_prediction_trials

# Same 6-channel order as detection/dataset.py -- see the "KNOWN GAP"
# note above for why this isn't Euler-inclusive despite the blueprint.
CHANNELS = ACCEL_COLUMNS + GYRO_COLUMNS


@dataclass
class WindowRecord:
    dataset: str             # always "kfall" -- kept as a column (rather than assumed) so downstream code that also touches detection's window records can share column-handling logic without a dataset-specific branch.
    subject_id: str
    global_subject_id: str
    activity_code: str
    trial_id: str
    label: str                # one of prediction.labelers.{NON_FALL,PRE_IMPACT,FALL}
    label_id: int              # LABEL_TO_INT[label] -- precomputed so training code doesn't need to re-import labelers.py just to get an integer target
    window_index: int         # 0-based index of this window within its source trial
    start_frame: int
    end_frame: int             # exclusive, BEFORE padding
    n_real_samples: int
    n_pad_samples: int
    harmonized_path: str


def build_windows_manifest(
    trial_manifest_df: pd.DataFrame,
    config: Optional[PredictionWindowingConfig] = None,
) -> pd.DataFrame:
    """Build the window-level manifest for the prediction pipeline.

    `trial_manifest_df` is the trial-level manifest from
    `shared.manifest.load_manifest` -- filtered here via
    `query_prediction_trials` (KFall-only, onset/impact-eligible trials
    only; see that function's docstring) and expanded into dense window
    boundaries, each labeled via `onset_impact_label`.
    """
    config = config or PredictionWindowingConfig()
    prediction_trials = query_prediction_trials(trial_manifest_df)

    records: list[WindowRecord] = []
    for _, row in prediction_trials.iterrows():
        n_samples = round(row["duration_s"] * row["sample_rate_hz"])
        window_specs = generate_window_specs(n_samples, config)

        onset_frame = row["fall_onset_frame"]
        impact_frame = row["fall_impact_frame"]
        # Manifest columns come from a parquet round-trip, so a missing
        # value may arrive as float `nan` rather than Python `None` --
        # normalize before handing to onset_impact_label, which checks
        # `is None` specifically (see that function's docstring).
        onset_frame = None if pd.isna(onset_frame) else int(onset_frame)
        impact_frame = None if pd.isna(impact_frame) else int(impact_frame)

        global_subject_id = f'{row["dataset"]}_{row["subject_id"]}'

        for window_index, spec in enumerate(window_specs):
            label = onset_impact_label(
                spec.start_frame, spec.end_frame, onset_frame, impact_frame
            )
            records.append(WindowRecord(
                dataset=row["dataset"],
                subject_id=row["subject_id"],
                global_subject_id=global_subject_id,
                activity_code=row["activity_code"],
                trial_id=row["trial_id"],
                label=label,
                label_id=LABEL_TO_INT[label],
                window_index=window_index,
                start_frame=spec.start_frame,
                end_frame=spec.end_frame,
                n_real_samples=spec.n_real_samples,
                n_pad_samples=spec.n_pad_samples,
                harmonized_path=row["harmonized_path"],
            ))

    if not records:
        return pd.DataFrame([asdict(r) for r in [_EMPTY_RECORD]]).iloc[0:0]

    return pd.DataFrame([asdict(r) for r in records])


_EMPTY_RECORD = WindowRecord(
    dataset="", subject_id="", global_subject_id="", activity_code="", trial_id="",
    label="", label_id=0, window_index=0, start_frame=0, end_frame=0,
    n_real_samples=0, n_pad_samples=0, harmonized_path="",
)


def load_window(
    window_row: pd.Series,
    window_length_samples: int,
    signal_cache: Optional[dict[str, pd.DataFrame]] = None,
) -> np.ndarray:
    """Load one window's actual signal data as a
    (window_length_samples, 6) array in CHANNELS order.

    Identical edge-padding strategy to `detection.dataset.load_window`
    (repeat the last real sample, not zero-pad -- see that function's
    docstring for the physical-plausibility rationale, which applies
    unchanged here). Duplicated rather than imported from
    `detection.dataset` per the blueprint's no-cross-import rule
    between the two pipelines; kept in sync by both being covered by
    their own pipeline's tests.
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
        raise ValueError(
            f"Window has 0 real samples (path={path}, start={start}, end={end}) "
            "-- cannot edge-pad from nothing."
        )

    pad_block = np.repeat(real_segment[-1:], n_pad, axis=0)
    return np.concatenate([real_segment, pad_block], axis=0)
