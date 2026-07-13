"""Per-window label assignment for the prediction pipeline.

Kept as its own module (not folded into `dataset.py`), matching the
blueprint's explicit call-out: "Label functions (`whole_trial_label`
vs `onset_impact_label`) -- different semantics, kept as separate
functions ... so each is independently testable."

Three-class scheme, per blueprint Pipeline 2 §4 (the standard framing
in the published pre-impact literature):
  - `non_fall`:    window entirely before the onset frame, or entirely
                    within an ADL trial (no onset/impact at all).
  - `pre_impact`:  window overlapping the onset->impact interval.
  - `fall`:        window overlapping or after the impact frame.

A binary collapse (pre-impact vs. not) is mentioned in the blueprint as
a fallback "if the three-class boundary proves too noisy in practice
-- decide after inspecting the per-class confusion matrix on a first
pass, not up front." So: NOT implemented here yet, on purpose --
`LABEL_TO_INT` stays 3-class until a real first-pass confusion matrix
(on real KFall data) actually motivates collapsing it, rather than
pre-deciding that from first principles.
"""
from __future__ import annotations

from typing import Optional

NON_FALL = "non_fall"
PRE_IMPACT = "pre_impact"
FALL = "fall"

LABEL_TO_INT = {NON_FALL: 0, PRE_IMPACT: 1, FALL: 2}


def onset_impact_label(
    start_frame: int,
    end_frame: int,
    onset_frame: Optional[int],
    impact_frame: Optional[int],
) -> str:
    """Label one window given its frame range and the source trial's
    onset/impact frames.

    `start_frame`/`end_frame` follow `shared.windowing.WindowSpec`'s
    convention: end-exclusive, so the window's real frames are
    `[start_frame, end_frame - 1]`.

    `onset_frame`/`impact_frame` are `None` for ADL trials (by
    definition -- an ADL trial has no fall event) -- always `non_fall`
    in that case, regardless of frame range. This function does NOT
    itself decide which trials are eligible to be labeled at all
    (e.g. a fall trial with a missing/unlabeled onset frame) -- that
    filtering already happens upstream in
    `shared.manifest.query_prediction_trials`, so by the time a fall
    trial's windows reach this function, onset/impact are expected to
    both be non-None. Still validated defensively below rather than
    silently mis-labeling if that upstream contract is ever violated.

    Precedence when a window's frame range overlaps BOTH the
    onset->impact interval AND extends to/past the impact frame (a
    real possibility here, unlike detection's coarser windows -- a
    1.0s prediction window can be comparable in length to an entire
    ~0.6-1.0s KFall fall event, per the blueprint, so a single window
    can plausibly span onset through impact and beyond): `fall` wins.
    A window containing the actual impact is the more safety-relevant
    state to report distinctly, and this keeps the three classes
    mutually exclusive by construction (every window gets exactly one
    label) rather than needing a separate tie-breaking policy at
    dataset-construction time.
    """
    if onset_frame is None or impact_frame is None:
        return NON_FALL

    if onset_frame >= impact_frame:
        raise ValueError(
            f"onset_frame ({onset_frame}) must be strictly before "
            f"impact_frame ({impact_frame}) -- this should have been "
            f"caught by shared.harmonize.validation upstream; a trial "
            f"reaching the labeler with onset >= impact indicates a "
            f"contract violation, not a normal data case."
        )

    # Window's last real frame is (end_frame - 1). "Overlapping or
    # after impact_frame" == that last frame reaches impact_frame,
    # i.e. end_frame > impact_frame.
    if end_frame > impact_frame:
        return FALL

    # Falling through here guarantees end_frame <= impact_frame, so
    # the only remaining overlap check needed against the
    # onset->impact interval is against its start.
    if end_frame > onset_frame:
        return PRE_IMPACT

    return NON_FALL
