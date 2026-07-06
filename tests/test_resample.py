"""Tests for shared/harmonize/resample.py (Stage 3, Task 3.2)."""
import numpy as np
import pandas as pd
import pytest

from shared.harmonize.resample import resample_signal


def _make_sine_signal(freq_hz: float, duration_s: float, sample_rate_hz: float, phase: float = 0.0) -> pd.DataFrame:
    n = int(duration_s * sample_rate_hz)
    t = np.arange(n) / sample_rate_hz
    values = np.sin(2 * np.pi * freq_hz * t + phase)
    return pd.DataFrame({"time_s": t, "signal": values})


def _dominant_frequency(values: np.ndarray, sample_rate_hz: float) -> float:
    n = len(values)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    magnitude = np.abs(np.fft.rfft(values))
    # ignore the DC bin (index 0) when finding the dominant frequency
    return freqs[1:][np.argmax(magnitude[1:])]


def test_noop_when_rates_equal():
    signal = _make_sine_signal(freq_hz=5, duration_s=2, sample_rate_hz=100)
    out = resample_signal(signal, native_rate_hz=100, target_rate_hz=100)

    pd.testing.assert_frame_equal(out, signal)


def test_noop_returns_new_object_not_same_reference():
    signal = _make_sine_signal(freq_hz=5, duration_s=1, sample_rate_hz=100)
    out = resample_signal(signal, native_rate_hz=100, target_rate_hz=100)
    assert out is not signal


def test_upsampling_raises_value_error():
    signal = _make_sine_signal(freq_hz=5, duration_s=1, sample_rate_hz=100)
    with pytest.raises(ValueError, match="downsampling"):
        resample_signal(signal, native_rate_hz=100, target_rate_hz=200)


def test_downsample_preserves_low_frequency_content():
    # A 5 Hz sine at 200 Hz, downsampled to 100 Hz, should still show a
    # dominant frequency close to 5 Hz -- well below both the old and
    # new Nyquist frequencies, so nothing should distort it.
    signal = _make_sine_signal(freq_hz=5, duration_s=4, sample_rate_hz=200)
    out = resample_signal(signal, native_rate_hz=200, target_rate_hz=100)

    dominant = _dominant_frequency(out["signal"].to_numpy(), sample_rate_hz=100)
    assert abs(dominant - 5) < 0.5


def test_downsample_preserves_amplitude_for_low_frequency_content():
    signal = _make_sine_signal(freq_hz=5, duration_s=4, sample_rate_hz=200)
    out = resample_signal(signal, native_rate_hz=200, target_rate_hz=100)

    original_amplitude = signal["signal"].abs().max()
    resampled_amplitude = out["signal"].abs().max()
    assert abs(original_amplitude - resampled_amplitude) < 0.15


def test_downsample_anti_aliases_high_frequency_content():
    # An 80 Hz sine at 200 Hz has content above the new 50 Hz Nyquist
    # frequency once resampled to 100 Hz. Without anti-aliasing, this
    # would fold back into a strong spurious 20 Hz component
    # (|100 - 80| = 20). With resample_poly's built-in anti-alias
    # filter, that 20 Hz alias should NOT dominate the output.
    signal = _make_sine_signal(freq_hz=80, duration_s=4, sample_rate_hz=200)
    out = resample_signal(signal, native_rate_hz=200, target_rate_hz=100)

    freqs = np.fft.rfftfreq(len(out), d=1.0 / 100)
    magnitude = np.abs(np.fft.rfft(out["signal"].to_numpy()))

    alias_bin = np.argmin(np.abs(freqs - 20))
    total_energy = magnitude.sum()
    alias_energy_fraction = magnitude[alias_bin] / total_energy

    # The 20 Hz alias bin should carry only a small fraction of total
    # energy -- most energy should have been suppressed by the
    # anti-alias filter rather than folded into this bin.
    assert alias_energy_fraction < 0.15


def test_output_timestamps_evenly_spaced_at_target_rate():
    signal = _make_sine_signal(freq_hz=5, duration_s=3, sample_rate_hz=200)
    out = resample_signal(signal, native_rate_hz=200, target_rate_hz=100)

    diffs = np.diff(out["time_s"].to_numpy())
    assert np.allclose(diffs, 1.0 / 100, atol=1e-9)


def test_non_integer_rate_ratio_is_supported():
    # Simulates a FallAllD-like non-integer-ratio rate (e.g. ~238 Hz
    # native) downsampled to the common 100 Hz target.
    signal = _make_sine_signal(freq_hz=5, duration_s=2, sample_rate_hz=238)
    out = resample_signal(signal, native_rate_hz=238, target_rate_hz=100)

    expected_n = int(round(len(signal) * 100 / 238))
    assert abs(len(out) - expected_n) <= 1

    dominant = _dominant_frequency(out["signal"].to_numpy(), sample_rate_hz=100)
    assert abs(dominant - 5) < 0.5


def test_multiple_columns_resampled_independently():
    n = 400
    t = np.arange(n) / 200
    df = pd.DataFrame({
        "time_s": t,
        "acc_x": np.sin(2 * np.pi * 5 * t),
        "acc_y": np.cos(2 * np.pi * 3 * t),
    })
    out = resample_signal(df, native_rate_hz=200, target_rate_hz=100)

    assert set(out.columns) == {"time_s", "acc_x", "acc_y"}
    assert abs(_dominant_frequency(out["acc_x"].to_numpy(), 100) - 5) < 0.5
    assert abs(_dominant_frequency(out["acc_y"].to_numpy(), 100) - 3) < 0.5
