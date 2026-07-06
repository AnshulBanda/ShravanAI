"""Tests for shared/harmonize/filtering.py (Stage 3, Task 3.3)."""
import numpy as np
import pandas as pd
import pytest

from shared.harmonize.filtering import apply_bandpass_filter

SAMPLE_RATE_HZ = 100.0


def _time_axis(duration_s: float, sample_rate_hz: float = SAMPLE_RATE_HZ) -> np.ndarray:
    return np.arange(int(duration_s * sample_rate_hz)) / sample_rate_hz


def test_composite_signal_drift_removed_movement_preserved_noise_removed():
    t = _time_axis(10)
    drift = 0.5 * np.sin(2 * np.pi * 0.1 * t)      # slow postural drift, below 0.5 Hz band edge
    movement = 1.0 * np.sin(2 * np.pi * 2.0 * t)   # genuine body movement, well inside 0.5-20 Hz
    noise = 0.3 * np.sin(2 * np.pi * 30.0 * t)     # high-frequency noise, above 20 Hz band edge
    composite = drift + movement + noise

    signal = pd.DataFrame({"time_s": t, "value": composite})
    out = apply_bandpass_filter(signal, columns=["value"], sample_rate_hz=SAMPLE_RATE_HZ)

    # Use only the interior of the signal to avoid filtfilt edge transients
    interior = slice(100, -100)
    out_vals = out["value"].to_numpy()[interior]
    movement_only = movement[interior]

    # movement component's amplitude should be largely preserved
    assert np.corrcoef(out_vals, movement_only)[0, 1] > 0.9

    # overall filtered amplitude should be much closer to the movement
    # component alone than to the full composite (drift + noise mostly gone)
    assert np.std(out_vals - movement_only) < np.std(composite[interior] - movement_only)


def test_impact_spike_survives_filtering():
    # The empirical justification for 0.5-20 Hz over the originally
    # proposed 5 Hz cutoff: a fall-impact-like transient must still be
    # clearly visible after filtering, or the cutoff choice is wrong.
    duration_s = 3.0
    t = _time_axis(duration_s)
    background_noise = 0.05 * np.random.default_rng(0).normal(size=len(t))

    pulse_start_s = 1.5
    pulse_duration_s = 0.1  # 100ms half-sine, mimicking a fall impact
    pulse_amplitude = 5.0

    signal_vals = background_noise.copy()
    pulse_start_idx = int(pulse_start_s * SAMPLE_RATE_HZ)
    pulse_n_samples = int(pulse_duration_s * SAMPLE_RATE_HZ)
    pulse_t = np.linspace(0, np.pi, pulse_n_samples)
    signal_vals[pulse_start_idx : pulse_start_idx + pulse_n_samples] += pulse_amplitude * np.sin(pulse_t)

    signal = pd.DataFrame({"time_s": t, "value": signal_vals})
    out = apply_bandpass_filter(signal, columns=["value"], sample_rate_hz=SAMPLE_RATE_HZ)

    original_peak = signal_vals.max()
    filtered_peak = out["value"].to_numpy().max()

    retention_fraction = filtered_peak / original_peak
    assert retention_fraction > 0.7, (
        f"Impact spike retained only {retention_fraction:.1%} of original amplitude "
        f"after filtering -- the 0.5-20 Hz cutoff may be too aggressive."
    )


def test_dc_constant_signal_heavily_attenuated():
    t = _time_axis(5)
    constant = np.full_like(t, 3.0)
    signal = pd.DataFrame({"time_s": t, "value": constant})

    out = apply_bandpass_filter(signal, columns=["value"], sample_rate_hz=SAMPLE_RATE_HZ)

    interior = out["value"].to_numpy()[100:-100]
    assert np.abs(interior).max() < 0.5  # far from the original constant value of 3.0


def test_filter_is_zero_phase_no_time_shift():
    # A symmetric pulse centered in the signal should stay centered
    # after filtering -- filtfilt guarantees zero phase distortion.
    t = _time_axis(4)
    center_idx = len(t) // 2
    signal_vals = np.zeros_like(t)
    pulse_width = 10
    pulse_t = np.linspace(0, np.pi, pulse_width)
    signal_vals[center_idx - pulse_width // 2 : center_idx - pulse_width // 2 + pulse_width] = np.sin(pulse_t)

    signal = pd.DataFrame({"time_s": t, "value": signal_vals})
    out = apply_bandpass_filter(signal, columns=["value"], sample_rate_hz=SAMPLE_RATE_HZ)

    original_peak_idx = np.argmax(signal_vals)
    filtered_peak_idx = np.argmax(out["value"].to_numpy())

    assert abs(original_peak_idx - filtered_peak_idx) <= 2


def test_columns_not_listed_are_passed_through_unchanged():
    t = _time_axis(2)
    signal = pd.DataFrame({
        "time_s": t,
        "acc_x": np.sin(2 * np.pi * 2 * t),
        "label_passthrough": np.arange(len(t)),
    })
    out = apply_bandpass_filter(signal, columns=["acc_x"], sample_rate_hz=SAMPLE_RATE_HZ)

    pd.testing.assert_series_equal(out["label_passthrough"], signal["label_passthrough"])


def test_high_hz_above_nyquist_raises_value_error():
    t = _time_axis(1, sample_rate_hz=30)
    signal = pd.DataFrame({"time_s": t, "value": np.sin(2 * np.pi * 2 * t)})

    with pytest.raises(ValueError, match="Nyquist"):
        apply_bandpass_filter(signal, columns=["value"], sample_rate_hz=30, high_hz=20)


def test_does_not_mutate_input():
    t = _time_axis(2)
    signal = pd.DataFrame({"time_s": t, "value": np.sin(2 * np.pi * 2 * t)})
    original = signal.copy()

    _ = apply_bandpass_filter(signal, columns=["value"], sample_rate_hz=SAMPLE_RATE_HZ)

    pd.testing.assert_frame_equal(signal, original)
