"""Tests for shared/harmonize/validation.py (Stage 3, Task 3.8)."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from shared.harmonize.axis_alignment import CalibrationResult
from shared.harmonize.validation import validate_harmonized_trial

SAMPLE_RATE_HZ = 100.0


@dataclass
class _FakeMetadata:
    label: str = "adl"
    fall_onset_frame: Optional[int] = None
    fall_impact_frame: Optional[int] = None


def _clean_signal(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    t = np.arange(n) / SAMPLE_RATE_HZ
    return pd.DataFrame({
        "time_s": t,
        "acc_x": rng.normal(0, 0.1, n), "acc_y": rng.normal(0, 0.1, n), "acc_z": rng.normal(0, 0.1, n),
        "gyro_x": rng.normal(0, 1.0, n), "gyro_y": rng.normal(0, 1.0, n), "gyro_z": rng.normal(0, 1.0, n),
    })


def _good_calibration() -> CalibrationResult:
    return CalibrationResult(rotation=np.eye(3), source="T01", gravity_vector=np.array([0.0, 0.0, 1.0]))


def test_clean_trial_produces_no_issues():
    issues = validate_harmonized_trial(
        _clean_signal(), _FakeMetadata(), _good_calibration(), expected_rate_hz=SAMPLE_RATE_HZ
    )
    assert issues == []


def test_schema_mismatch_detected():
    signal = _clean_signal().drop(columns=["gyro_z"])
    issues = validate_harmonized_trial(
        signal, _FakeMetadata(), _good_calibration(), expected_rate_hz=SAMPLE_RATE_HZ
    )
    assert any("Schema mismatch" in i for i in issues)


def test_nan_detected():
    signal = _clean_signal()
    signal.loc[10, "acc_x"] = np.nan
    issues = validate_harmonized_trial(
        signal, _FakeMetadata(), _good_calibration(), expected_rate_hz=SAMPLE_RATE_HZ
    )
    assert any("NaN" in i for i in issues)


def test_infinite_value_detected():
    signal = _clean_signal()
    signal.loc[10, "acc_x"] = np.inf
    issues = validate_harmonized_trial(
        signal, _FakeMetadata(), _good_calibration(), expected_rate_hz=SAMPLE_RATE_HZ
    )
    assert any("Infinite" in i for i in issues)


def test_uneven_timestamp_spacing_detected():
    signal = _clean_signal()
    times = signal["time_s"].to_numpy().copy()
    times[150] += 0.5  # inject a large jump
    signal["time_s"] = times
    issues = validate_harmonized_trial(
        signal, _FakeMetadata(), _good_calibration(), expected_rate_hz=SAMPLE_RATE_HZ
    )
    assert any("timestamp spacing" in i for i in issues)


def test_implausible_acceleration_magnitude_detected():
    signal = _clean_signal()
    signal.loc[10, "acc_x"] = 50.0  # far beyond any real hardware range
    issues = validate_harmonized_trial(
        signal, _FakeMetadata(), _good_calibration(), expected_rate_hz=SAMPLE_RATE_HZ
    )
    assert any("implausible acceleration" in i for i in issues)


def test_flatline_detected():
    signal = _clean_signal()
    signal["acc_x"] = 0.0  # zero variance across the whole trial
    issues = validate_harmonized_trial(
        signal, _FakeMetadata(), _good_calibration(), expected_rate_hz=SAMPLE_RATE_HZ
    )
    assert any("flatline" in i for i in issues)


def test_calibration_sanity_failure_detected():
    # A rotation that does NOT align gravity to vertical -- e.g. gravity
    # still on the y-axis after "calibration".
    bad_calibration = CalibrationResult(
        rotation=np.eye(3), source="T01", gravity_vector=np.array([0.0, -1.0, 0.0])
    )
    issues = validate_harmonized_trial(
        _clean_signal(), _FakeMetadata(), bad_calibration, expected_rate_hz=SAMPLE_RATE_HZ
    )
    assert any("Calibration sanity failed" in i for i in issues)


def test_duration_outside_expected_range_detected():
    signal = _clean_signal(n=100)  # 1.0s at 100Hz
    issues = validate_harmonized_trial(
        signal, _FakeMetadata(), _good_calibration(), expected_rate_hz=SAMPLE_RATE_HZ,
        expected_duration_range_s=(25.0, 35.0),  # e.g. T01's expected ~30s
    )
    assert any("duration" in i for i in issues)


def test_duration_check_skipped_when_range_not_provided():
    signal = _clean_signal(n=100)
    issues = validate_harmonized_trial(
        signal, _FakeMetadata(), _good_calibration(), expected_rate_hz=SAMPLE_RATE_HZ
    )
    assert not any("duration" in i for i in issues)


def test_adl_with_nonnull_onset_impact_detected():
    metadata = _FakeMetadata(label="adl", fall_onset_frame=50, fall_impact_frame=80)
    issues = validate_harmonized_trial(
        _clean_signal(), metadata, _good_calibration(), expected_rate_hz=SAMPLE_RATE_HZ
    )
    assert any("ADL trial has non-null" in i for i in issues)


def test_fall_onset_after_impact_detected():
    metadata = _FakeMetadata(label="fall", fall_onset_frame=200, fall_impact_frame=100)
    issues = validate_harmonized_trial(
        _clean_signal(), metadata, _good_calibration(), expected_rate_hz=SAMPLE_RATE_HZ
    )
    assert any("not before impact_frame" in i for i in issues)


def test_fall_impact_beyond_signal_length_detected():
    metadata = _FakeMetadata(label="fall", fall_onset_frame=100, fall_impact_frame=5000)
    issues = validate_harmonized_trial(
        _clean_signal(), metadata, _good_calibration(), expected_rate_hz=SAMPLE_RATE_HZ
    )
    assert any("exceeds signal length" in i for i in issues)


def test_fall_with_valid_onset_impact_passes():
    metadata = _FakeMetadata(label="fall", fall_onset_frame=100, fall_impact_frame=150)
    issues = validate_harmonized_trial(
        _clean_signal(), metadata, _good_calibration(), expected_rate_hz=SAMPLE_RATE_HZ
    )
    assert issues == []


def test_metadata_without_label_fields_does_not_crash():
    # Simulates a dataset (e.g. SisFall) whose metadata object might not
    # have fall_onset_frame/fall_impact_frame/label at all.
    class _MinimalMetadata:
        pass

    issues = validate_harmonized_trial(
        _clean_signal(), _MinimalMetadata(), _good_calibration(), expected_rate_hz=SAMPLE_RATE_HZ
    )
    assert issues == []
