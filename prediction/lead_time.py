"""Lead-time metric for the prediction pipeline.

Per blueprint Pipeline 2 §7: "Report lead time as a first-class metric
alongside sensitivity/specificity: for every correctly flagged fall,
how many milliseconds before the impact frame did the model first
raise pre-impact. This is the metric the whole pipeline exists to
optimize, and it's absent from plain classification accuracy."

Operates per FALL TRIAL (not per window) -- takes one trial's windows,
already sorted by time, with the model's predicted label per window,
and asks: at what frame did the model FIRST predict `pre_impact`,
strictly before the real impact frame? That's the actual quantity of
interest; per-window classification accuracy alone doesn't capture it
(a model could get every individual window's 3-class label "right" on
average yet still flag the fall too late to be clinically useful, or
flag it early but only intermittently).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from prediction.labelers import LABEL_TO_INT, PRE_IMPACT

_PRE_IMPACT_ID = LABEL_TO_INT[PRE_IMPACT]


def compute_lead_time_ms(
    start_frames: Sequence[int],
    predicted_label_ids: Sequence[int],
    impact_frame: int,
    sample_rate_hz: float = 100.0,
) -> Optional[float]:
    """Milliseconds before `impact_frame` that the model FIRST
    predicted `pre_impact`, or `None` if it never did (for any window
    whose `start_frame` is strictly before `impact_frame`) -- a missed
    flag, for lead-time purposes specifically. This is a real,
    reportable outcome (not an error) -- callers computing an aggregate
    lead time across many trials need to handle `None` explicitly
    (see `summarize_lead_times` below) rather than treating it as a
    0ms lead time, which would silently and wrongly reward a model
    that never gives any advance warning at all.

    A window predicted `pre_impact` that starts AT OR AFTER
    `impact_frame` doesn't count -- that's not advance warning, the
    fall has already happened by then (matches
    `prediction.labelers.onset_impact_label`'s own boundary: `fall`
    takes precedence over `pre_impact` once a window reaches the
    impact frame, so in practice a well-trained model predicting
    `pre_impact` post-impact would itself be a misclassification --
    but this function doesn't assume the model behaves correctly, and
    excludes such windows regardless of why they occurred).

    `start_frames`/`predicted_label_ids` don't need to already be
    sorted by time -- sorted internally, defensively, since callers
    may hand this a DataFrame slice in whatever row order it happens
    to be in.
    """
    start_frames = np.asarray(start_frames)
    predicted_label_ids = np.asarray(predicted_label_ids)
    if len(start_frames) != len(predicted_label_ids):
        raise ValueError(
            f"start_frames ({len(start_frames)}) and predicted_label_ids "
            f"({len(predicted_label_ids)}) must be the same length -- one "
            f"prediction per window."
        )

    order = np.argsort(start_frames)
    start_frames = start_frames[order]
    predicted_label_ids = predicted_label_ids[order]

    eligible = (start_frames < impact_frame) & (predicted_label_ids == _PRE_IMPACT_ID)
    if not np.any(eligible):
        return None

    first_flagged_frame = start_frames[eligible][0]  # earliest, since already sorted
    return float((impact_frame - first_flagged_frame) / sample_rate_hz * 1000.0)


@dataclass
class LeadTimeSummary:
    n_trials: int
    n_flagged: int              # trials where compute_lead_time_ms returned a real value, not None
    detection_rate: float       # n_flagged / n_trials -- fraction of falls given ANY advance warning at all
    mean_lead_time_ms: Optional[float]
    median_lead_time_ms: Optional[float]


def summarize_lead_times(lead_times: Sequence[Optional[float]]) -> LeadTimeSummary:
    """Aggregate per-trial lead times (as returned by
    `compute_lead_time_ms`, one entry per fall trial, `None` entries
    included) into trial-level summary statistics.

    `mean_lead_time_ms`/`median_lead_time_ms` are computed over the
    FLAGGED trials only (`None`s excluded) -- reporting a mean lead
    time is only meaningful conditional on the fall having been
    flagged at all; `detection_rate` is the separate, equally
    important number for how often that condition holds in the first
    place. Reporting only a mean-including-zeros-for-misses would
    conflate "flagged late" with "never flagged" under one number,
    losing exactly the distinction blueprint §7 asks this metric to
    surface.
    """
    n_trials = len(lead_times)
    if n_trials == 0:
        raise ValueError("lead_times is empty -- nothing to summarize.")

    flagged = [lt for lt in lead_times if lt is not None]
    n_flagged = len(flagged)

    return LeadTimeSummary(
        n_trials=n_trials,
        n_flagged=n_flagged,
        detection_rate=n_flagged / n_trials,
        mean_lead_time_ms=float(np.mean(flagged)) if flagged else None,
        median_lead_time_ms=float(np.median(flagged)) if flagged else None,
    )
