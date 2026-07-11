"""Tests for detection/windowing.py -- pure window-boundary logic."""
from detection.windowing import WindowingConfig, generate_window_specs


def _config():
    # 200-sample windows, 100-sample stride, matching the blueprint's
    # 2.0s window / 1.0s stride @ 100Hz spec.
    return WindowingConfig(window_length_s=2.0, stride_s=1.0, target_rate_hz=100.0)


def test_windowing_config_derives_correct_sample_counts():
    config = _config()
    assert config.window_length_samples == 200
    assert config.stride_samples == 100


def test_trial_shorter_than_one_window_produces_single_padded_window():
    specs = generate_window_specs(trial_n_samples=120, config=_config())

    assert len(specs) == 1
    assert specs[0].start_frame == 0
    assert specs[0].end_frame == 120
    assert specs[0].n_real_samples == 120
    assert specs[0].n_pad_samples == 80  # 200 - 120


def test_trial_exactly_one_window_length_produces_no_padding():
    specs = generate_window_specs(trial_n_samples=200, config=_config())

    assert len(specs) == 1
    assert specs[0].n_real_samples == 200
    assert specs[0].n_pad_samples == 0


def test_trial_with_clean_multiple_of_stride_has_no_trailing_window():
    # 500 samples: full windows at start=0 (0-200), 100 (100-300), 200
    # (200-400), 300 (300-500) -- exactly covers to the end, no leftover.
    specs = generate_window_specs(trial_n_samples=500, config=_config())

    assert len(specs) == 4
    assert [s.start_frame for s in specs] == [0, 100, 200, 300]
    assert all(s.n_pad_samples == 0 for s in specs)
    assert specs[-1].end_frame == 500


def test_trial_with_leftover_gets_one_trailing_padded_window():
    # 550 samples: full windows at 0,100,200,300 (covering to 500, the
    # last one ending exactly at 500), then 50 samples left over
    # (500-550) -- must NOT be dropped, and must NOT overlap the last
    # full window (starts at 500, where the last one ended, not at the
    # next stride position 400).
    specs = generate_window_specs(trial_n_samples=550, config=_config())

    assert len(specs) == 5
    full_windows, trailing = specs[:4], specs[4]
    assert all(s.n_pad_samples == 0 for s in full_windows)
    assert trailing.start_frame == 500
    assert trailing.end_frame == 550
    assert trailing.n_real_samples == 50
    assert trailing.n_pad_samples == 150


def test_windows_never_cross_or_exceed_trial_boundary():
    specs = generate_window_specs(trial_n_samples=337, config=_config())

    for s in specs:
        assert s.end_frame <= 337
        assert s.start_frame >= 0
        assert s.end_frame - s.start_frame == s.n_real_samples


def test_entire_trial_is_covered_start_to_end():
    # No gaps: every sample index from 0 to trial_n_samples-1 falls
    # inside at least one window.
    n = 733
    specs = generate_window_specs(trial_n_samples=n, config=_config())

    covered = set()
    for s in specs:
        covered.update(range(s.start_frame, s.end_frame))
    assert covered == set(range(n))


def test_zero_or_negative_trial_length_returns_no_windows():
    assert generate_window_specs(0, _config()) == []
    assert generate_window_specs(-5, _config()) == []


def test_window_length_samples_never_exceeded():
    for n in [1, 50, 199, 200, 201, 350, 999, 1000, 1001]:
        specs = generate_window_specs(n, _config())
        for s in specs:
            assert s.n_real_samples + s.n_pad_samples == 200
