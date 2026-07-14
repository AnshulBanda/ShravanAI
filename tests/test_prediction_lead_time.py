"""Tests for prediction/lead_time.py."""
import pytest

from prediction.labelers import LABEL_TO_INT, NON_FALL, PRE_IMPACT, FALL
from prediction.lead_time import compute_lead_time_ms, summarize_lead_times

NF, PI, FL = LABEL_TO_INT[NON_FALL], LABEL_TO_INT[PRE_IMPACT], LABEL_TO_INT[FALL]


def test_lead_time_computed_from_first_pre_impact_flag():
    # Windows at frames 0,10,...,90; impact at frame 100 (10-frame
    # windows here for simplicity, not the real 100-sample windows).
    start_frames = [0, 10, 20, 30, 40, 50]
    predicted = [NF, NF, PI, PI, PI, FL]  # first pre_impact at frame 20

    lead_ms = compute_lead_time_ms(start_frames, predicted, impact_frame=100, sample_rate_hz=100.0)

    # (100 - 20) / 100 * 1000 = 800ms
    assert lead_ms == pytest.approx(800.0)


def test_never_flagged_before_impact_returns_none():
    start_frames = [0, 10, 20, 30]
    predicted = [NF, NF, NF, NF]  # never predicted pre_impact at all

    lead_ms = compute_lead_time_ms(start_frames, predicted, impact_frame=100)

    assert lead_ms is None


def test_pre_impact_flag_at_or_after_impact_does_not_count():
    # Only "flags" pre_impact once impact has already happened -- not
    # real advance warning.
    start_frames = [90, 100, 110]
    predicted = [NF, PI, PI]  # frame 100 == impact_frame, doesn't count

    lead_ms = compute_lead_time_ms(start_frames, predicted, impact_frame=100)

    assert lead_ms is None


def test_unsorted_input_handled_correctly():
    # Same data as the first test, but shuffled -- must sort internally.
    start_frames = [50, 0, 30, 20, 10, 40]
    predicted = [FL, NF, PI, PI, NF, PI]

    lead_ms = compute_lead_time_ms(start_frames, predicted, impact_frame=100, sample_rate_hz=100.0)

    assert lead_ms == pytest.approx(800.0)  # first real pre_impact still at frame 20


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        compute_lead_time_ms([0, 10], [NF, NF, PI], impact_frame=100)


def test_takes_earliest_pre_impact_flag_not_latest():
    start_frames = [0, 10, 20, 30, 40]
    predicted = [NF, PI, NF, PI, FL]  # pre_impact at 10 AND 30 -- should use 10 (earliest)

    lead_ms = compute_lead_time_ms(start_frames, predicted, impact_frame=100, sample_rate_hz=100.0)

    assert lead_ms == pytest.approx(900.0)  # (100-10)/100*1000


# --- summarize_lead_times ---

def test_summary_detection_rate_and_means():
    lead_times = [800.0, 900.0, None, 500.0, None]

    summary = summarize_lead_times(lead_times)

    assert summary.n_trials == 5
    assert summary.n_flagged == 3
    assert summary.detection_rate == pytest.approx(0.6)
    assert summary.mean_lead_time_ms == pytest.approx((800 + 900 + 500) / 3)
    assert summary.median_lead_time_ms == pytest.approx(800.0)


def test_summary_all_missed():
    summary = summarize_lead_times([None, None, None])

    assert summary.detection_rate == 0.0
    assert summary.mean_lead_time_ms is None
    assert summary.median_lead_time_ms is None


def test_summary_all_flagged():
    summary = summarize_lead_times([100.0, 200.0, 300.0])

    assert summary.detection_rate == 1.0
    assert summary.mean_lead_time_ms == pytest.approx(200.0)


def test_summary_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        summarize_lead_times([])
