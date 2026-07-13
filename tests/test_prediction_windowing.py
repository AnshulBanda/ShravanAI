"""Tests for prediction/windowing.py -- dense-config defaults + reuse
of shared boundary logic (the boundary math itself is already covered
by tests/test_windowing.py against shared/windowing.py via detection's
re-export; this file only checks prediction's OWN defaults and that
dense (heavily-overlapping) configs behave correctly)."""
from prediction.windowing import PredictionWindowingConfig, generate_window_specs


def test_prediction_defaults_match_blueprint_pipeline_2_spec():
    config = PredictionWindowingConfig()
    assert config.window_length_s == 1.0
    assert config.stride_s == 0.1
    assert config.window_length_samples == 100
    assert config.stride_samples == 10


def test_dense_stride_produces_heavily_overlapping_windows():
    # 300 samples @ 100-sample window / 10-sample stride ->
    # windows start at 0,10,...,190 (last full window 190-290),
    # then one trailing padded window for the remaining 290-300.
    specs = generate_window_specs(300, PredictionWindowingConfig())

    full_windows = [s for s in specs if s.n_pad_samples == 0]
    assert len(full_windows) == 21  # (300-100)/10 + 1
    assert full_windows[0].start_frame == 0
    assert full_windows[-1].start_frame == 200
    assert full_windows[-1].end_frame == 300
    # 200 + 100 == 300 exactly -- covers to the trial end with no
    # leftover, so no trailing padded window here at all.
    assert len(specs) == 21
