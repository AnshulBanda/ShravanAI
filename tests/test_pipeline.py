"""Tests for shared/harmonize/pipeline.py (Stage 3, Task 3.7)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import shared.harmonize.pipeline as pipeline_module
from shared.harmonize.axis_alignment import CalibrationResult, compute_gravity_rotation
from shared.harmonize.pipeline import HarmonizationConfig, harmonize_trial

SAMPLE_RATE_HZ = 100.0


@dataclass
class _FakeMetadata:
    dataset: str
    native_rate_hz: float


@dataclass
class _FakeTrial:
    signal: pd.DataFrame
    metadata: _FakeMetadata


def _identity_calibration() -> CalibrationResult:
    return CalibrationResult(rotation=np.eye(3), source="T01", gravity_vector=np.array([0.0, 0.0, 1.0]))


def _dominant_frequency(values: np.ndarray, sample_rate_hz: float) -> float:
    n = len(values)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    magnitude = np.abs(np.fft.rfft(values))
    return freqs[1:][np.argmax(magnitude[1:])]


def test_output_schema_drops_extra_columns_like_euler():
    n = 300
    t = np.arange(n) / SAMPLE_RATE_HZ
    signal = pd.DataFrame({
        "time_s": t,
        "acc_x": np.zeros(n), "acc_y": np.zeros(n), "acc_z": np.ones(n),
        "gyro_x": np.zeros(n), "gyro_y": np.zeros(n), "gyro_z": np.zeros(n),
        "euler_x": np.zeros(n), "euler_y": np.zeros(n), "euler_z": np.zeros(n),
    })
    trial = _FakeTrial(signal=signal, metadata=_FakeMetadata(dataset="kfall", native_rate_hz=100.0))
    config = HarmonizationConfig()

    out = harmonize_trial(trial, _identity_calibration(), config)

    assert list(out.columns) == ["time_s", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    assert "euler_x" not in out.columns


def test_output_is_at_target_rate_after_downsampling():
    n = 400  # 4s at native 200 Hz -> should become 2s worth at 100 Hz -> 200 samples
    t = np.arange(n) / 200.0
    signal = pd.DataFrame({
        "time_s": t,
        "acc_x": np.sin(2 * np.pi * 3 * t), "acc_y": np.zeros(n), "acc_z": np.ones(n),
        "gyro_x": np.zeros(n), "gyro_y": np.zeros(n), "gyro_z": np.zeros(n),
    })
    trial = _FakeTrial(signal=signal, metadata=_FakeMetadata(dataset="kfall", native_rate_hz=200.0))
    config = HarmonizationConfig(target_rate_hz=100.0)

    out = harmonize_trial(trial, _identity_calibration(), config)

    assert abs(len(out) - 200) <= 1
    diffs = np.diff(out["time_s"].to_numpy())
    assert np.allclose(diffs, 1.0 / 100.0, atol=1e-6)


def test_impact_spike_survives_full_pipeline():
    n = 300
    t = np.arange(n) / SAMPLE_RATE_HZ
    rng = np.random.default_rng(0)
    accel_z = np.ones(n) + rng.normal(0, 0.01, n)
    pulse_start, pulse_n = 150, 10
    accel_z[pulse_start:pulse_start + pulse_n] += 5.0 * np.sin(np.linspace(0, np.pi, pulse_n))

    signal = pd.DataFrame({
        "time_s": t,
        "acc_x": rng.normal(0, 0.01, n), "acc_y": rng.normal(0, 0.01, n), "acc_z": accel_z,
        "gyro_x": rng.normal(0, 0.5, n), "gyro_y": rng.normal(0, 0.5, n), "gyro_z": rng.normal(0, 0.5, n),
    })
    trial = _FakeTrial(signal=signal, metadata=_FakeMetadata(dataset="kfall", native_rate_hz=100.0))
    config = HarmonizationConfig()

    out = harmonize_trial(trial, _identity_calibration(), config)

    # The spike itself (a transient ABOVE the removed near-DC gravity
    # bias) should still be clearly visible after the full pipeline,
    # even though the constant 1g bias around it is filtered away.
    interior = out["acc_z"].to_numpy()[20:-20]
    assert interior.max() > 2.0


def test_axis_alignment_effect_survives_filtering():
    # Gravity + a genuine 3 Hz movement oscillation both mismounted onto
    # acc_y (matching the real SA06 finding), with a rotation that maps
    # acc_y -> acc_z. The movement frequency (unlike the DC gravity
    # bias) survives the 0.5-20 Hz filter, so THIS is the correct way
    # to prove alignment actually took effect through the full pipeline
    # -- checking for a persistent ~1g bias would not work, since the
    # filter deliberately removes that DC content by design.
    n = 400
    t = np.arange(n) / SAMPLE_RATE_HZ
    rng = np.random.default_rng(1)
    movement = 0.3 * np.sin(2 * np.pi * 3 * t)

    signal = pd.DataFrame({
        "time_s": t,
        "acc_x": rng.normal(0, 0.01, n),
        "acc_y": -1.0 + movement + rng.normal(0, 0.01, n),  # gravity + movement, on acc_y
        "acc_z": rng.normal(0, 0.01, n),
        "gyro_x": rng.normal(0, 0.5, n), "gyro_y": rng.normal(0, 0.5, n), "gyro_z": rng.normal(0, 0.5, n),
    })
    trial = _FakeTrial(signal=signal, metadata=_FakeMetadata(dataset="kfall", native_rate_hz=100.0))

    accel_segment = signal[["acc_x", "acc_y", "acc_z"]].to_numpy()
    rotation = compute_gravity_rotation(accel_segment)
    calibration = CalibrationResult(rotation=rotation, source="T01", gravity_vector=accel_segment.mean(axis=0))

    out = harmonize_trial(trial, calibration, HarmonizationConfig())

    freq_on_z = _dominant_frequency(out["acc_z"].to_numpy(), SAMPLE_RATE_HZ)
    freq_on_y = _dominant_frequency(out["acc_y"].to_numpy(), SAMPLE_RATE_HZ)

    # After alignment, the 3 Hz movement should show up on acc_z, not acc_y.
    assert abs(freq_on_z - 3) < 0.5
    assert abs(freq_on_y - 3) > 0.5


def test_steps_run_in_the_frozen_order(monkeypatch):
    # Protects against an implementation-order bug specifically (the
    # four operations mathematically commute for a static rotation, per
    # PROJECT_CHECKPOINT.md, so this isn't testing a numerical
    # difference -- it's a direct regression check that the code calls
    # things in the intended sequence).
    call_order = []

    class _SpyConverter:
        def convert(self, signal):
            call_order.append("units")
            return signal

    def _spy_get_converter(dataset):
        return _SpyConverter()

    def _spy_resample(signal, native_rate_hz, target_rate_hz):
        call_order.append("resample")
        return signal

    def _spy_apply_rotation(signal, rotation):
        call_order.append("align")
        return signal

    def _spy_apply_bandpass(signal, columns, sample_rate_hz, low_hz, high_hz, order):
        call_order.append("filter")
        return signal

    monkeypatch.setattr(pipeline_module, "get_unit_converter", _spy_get_converter)
    monkeypatch.setattr(pipeline_module, "resample_signal", _spy_resample)
    monkeypatch.setattr(pipeline_module, "apply_rotation", _spy_apply_rotation)
    monkeypatch.setattr(pipeline_module, "apply_bandpass_filter", _spy_apply_bandpass)

    n = 10
    signal = pd.DataFrame({
        "time_s": np.arange(n) / 100.0,
        "acc_x": np.zeros(n), "acc_y": np.zeros(n), "acc_z": np.ones(n),
        "gyro_x": np.zeros(n), "gyro_y": np.zeros(n), "gyro_z": np.zeros(n),
    })
    trial = _FakeTrial(signal=signal, metadata=_FakeMetadata(dataset="kfall", native_rate_hz=100.0))

    harmonize_trial(trial, _identity_calibration(), HarmonizationConfig())

    assert call_order == ["units", "resample", "align", "filter"]
