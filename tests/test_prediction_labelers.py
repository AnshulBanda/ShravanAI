"""Tests for prediction/labelers.py -- 3-class onset/impact labeling."""
import pytest

from prediction.labelers import FALL, NON_FALL, PRE_IMPACT, onset_impact_label


def test_adl_trial_always_non_fall_regardless_of_frame_range():
    # onset_frame/impact_frame both None -- the ADL-trial signal.
    assert onset_impact_label(0, 100, None, None) == NON_FALL
    assert onset_impact_label(10_000, 10_100, None, None) == NON_FALL


def test_window_entirely_before_onset_is_non_fall():
    # onset=130, impact=208 (the project's real SA06 T22 R01 values,
    # per PROJECT_CHECKPOINT.md). A window ending well before onset.
    assert onset_impact_label(0, 100, onset_frame=130, impact_frame=208) == NON_FALL
    # Ends exactly AT onset (end_frame exclusive -> last real frame is
    # 129, still strictly before onset) -- boundary case, still non_fall.
    assert onset_impact_label(30, 130, onset_frame=130, impact_frame=208) == NON_FALL


def test_window_overlapping_onset_impact_interval_is_pre_impact():
    # Starts just before onset, ends before impact.
    assert onset_impact_label(125, 200, onset_frame=130, impact_frame=208) == PRE_IMPACT
    # Fully inside the onset->impact interval.
    assert onset_impact_label(140, 200, onset_frame=130, impact_frame=208) == PRE_IMPACT
    # Ends exactly at impact (last real frame 207, strictly before
    # impact=208) -- still pre_impact, not fall yet.
    assert onset_impact_label(140, 208, onset_frame=130, impact_frame=208) == PRE_IMPACT


def test_window_overlapping_or_after_impact_is_fall():
    # Overlaps impact frame itself (last real frame 208 >= impact=208).
    assert onset_impact_label(150, 209, onset_frame=130, impact_frame=208) == FALL
    # Entirely after impact.
    assert onset_impact_label(300, 400, onset_frame=130, impact_frame=208) == FALL


def test_window_spanning_onset_through_past_impact_resolves_to_fall_not_pre_impact():
    # A 1.0s (100-sample) window can be comparable to or longer than an
    # entire ~0.6-1.0s KFall fall event -- a real case per the blueprint,
    # not a hypothetical edge case. `fall` must win per the documented
    # precedence rule.
    assert onset_impact_label(100, 250, onset_frame=130, impact_frame=208) == FALL


def test_onset_not_before_impact_raises():
    with pytest.raises(ValueError, match="strictly before"):
        onset_impact_label(0, 100, onset_frame=208, impact_frame=130)
    with pytest.raises(ValueError, match="strictly before"):
        onset_impact_label(0, 100, onset_frame=100, impact_frame=100)
