"""Tests for shared/harmonize/units.py (Stage 3, Task 3.1 + Stage 5)."""
import numpy as np
import pandas as pd
import pytest

from shared.harmonize.units import (
    ACCEL_COLUMNS,
    GYRO_COLUMNS,
    KFallUnitConverter,
    SisFallUnitConverter,
    get_unit_converter,
)


def _sample_signal(n_rows: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "time_s": np.arange(n_rows) / 100.0,
            "acc_x": rng.normal(0, 0.1, n_rows),
            "acc_y": rng.normal(0, 0.1, n_rows),
            "acc_z": rng.normal(1.0, 0.1, n_rows),
            "gyro_x": rng.normal(0, 5.0, n_rows),
            "gyro_y": rng.normal(0, 5.0, n_rows),
            "gyro_z": rng.normal(0, 5.0, n_rows),
        }
    )


def _sample_raw_sisfall_signal(n_rows: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "time_s": np.arange(n_rows) / 200.0,
            "raw_adxl_acc_x": rng.integers(-2000, 2000, n_rows),
            "raw_adxl_acc_y": rng.integers(-2000, 2000, n_rows),
            "raw_adxl_acc_z": rng.integers(6000, 10000, n_rows),  # ~1g resting
            "raw_gyro_x": rng.integers(-500, 500, n_rows),
            "raw_gyro_y": rng.integers(-500, 500, n_rows),
            "raw_gyro_z": rng.integers(-500, 500, n_rows),
            "raw_mma_acc_x": rng.integers(-1000, 1000, n_rows),
            "raw_mma_acc_y": rng.integers(-1000, 1000, n_rows),
            "raw_mma_acc_z": rng.integers(3000, 5000, n_rows),
        }
    )


def test_kfall_converter_is_exact_noop():
    signal = _sample_signal()
    converted = KFallUnitConverter().convert(signal)

    pd.testing.assert_frame_equal(converted, signal)


def test_kfall_converter_does_not_mutate_input():
    signal = _sample_signal()
    original = signal.copy()

    _ = KFallUnitConverter().convert(signal)

    pd.testing.assert_frame_equal(signal, original)


def test_kfall_converter_returns_new_object_not_same_reference():
    signal = _sample_signal()
    converted = KFallUnitConverter().convert(signal)

    assert converted is not signal


def test_get_unit_converter_returns_kfall_converter():
    converter = get_unit_converter("kfall")
    assert isinstance(converter, KFallUnitConverter)


def test_get_unit_converter_case_insensitive():
    converter = get_unit_converter("KFall")
    assert isinstance(converter, KFallUnitConverter)


def test_get_unit_converter_unknown_dataset_raises_with_known_list():
    with pytest.raises(ValueError, match="kfall"):
        get_unit_converter("fallallD")


def test_expected_columns_defined_and_distinct():
    assert set(ACCEL_COLUMNS) == {"acc_x", "acc_y", "acc_z"}
    assert set(GYRO_COLUMNS) == {"gyro_x", "gyro_y", "gyro_z"}
    assert set(ACCEL_COLUMNS).isdisjoint(GYRO_COLUMNS)


# --- SisFall (Stage 5) ---

def test_get_unit_converter_returns_sisfall_converter():
    converter = get_unit_converter("sisfall")
    assert isinstance(converter, SisFallUnitConverter)


def test_sisfall_converter_produces_expected_columns():
    signal = _sample_raw_sisfall_signal()
    converted = SisFallUnitConverter().convert(signal)

    assert set(ACCEL_COLUMNS).issubset(converted.columns)
    assert set(GYRO_COLUMNS).issubset(converted.columns)
    assert {"mma_acc_x", "mma_acc_y", "mma_acc_z"}.issubset(converted.columns)


def test_sisfall_converter_uses_adxl345_as_primary_accelerometer():
    # A known raw ADXL345 value should convert via the documented
    # formula: physical = [(2*Range)/(2^Resolution)] * AD, Range=16,
    # Resolution=13 -- i.e. raw_adxl_acc_x=8192 (a clean value near
    # the 13-bit range) should convert to exactly 32.0g.
    signal = pd.DataFrame({
        "raw_adxl_acc_x": [8192], "raw_adxl_acc_y": [0], "raw_adxl_acc_z": [0],
        "raw_gyro_x": [0], "raw_gyro_y": [0], "raw_gyro_z": [0],
        "raw_mma_acc_x": [0], "raw_mma_acc_y": [0], "raw_mma_acc_z": [0],
    })
    converted = SisFallUnitConverter().convert(signal)

    assert converted["acc_x"].iloc[0] == pytest.approx(32.0)


def test_sisfall_converter_gyro_scale_matches_readme_formula():
    # ITG3200: Range=2000 deg/s, Resolution=16-bit.
    # raw_gyro_x=32768 -> [(2*2000)/(2**16)] * 32768 == 2000.0 deg/s
    signal = pd.DataFrame({
        "raw_adxl_acc_x": [0], "raw_adxl_acc_y": [0], "raw_adxl_acc_z": [0],
        "raw_gyro_x": [32768], "raw_gyro_y": [0], "raw_gyro_z": [0],
        "raw_mma_acc_x": [0], "raw_mma_acc_y": [0], "raw_mma_acc_z": [0],
    })
    converted = SisFallUnitConverter().convert(signal)

    assert converted["gyro_x"].iloc[0] == pytest.approx(2000.0)


def test_sisfall_converter_mma8451q_scale_matches_readme_formula():
    # MMA8451Q: Range=8g, Resolution=14-bit.
    # raw_mma_acc_x=8192 -> [(2*8)/(2**14)] * 8192 == 8.0g
    signal = pd.DataFrame({
        "raw_adxl_acc_x": [0], "raw_adxl_acc_y": [0], "raw_adxl_acc_z": [0],
        "raw_gyro_x": [0], "raw_gyro_y": [0], "raw_gyro_z": [0],
        "raw_mma_acc_x": [8192], "raw_mma_acc_y": [0], "raw_mma_acc_z": [0],
    })
    converted = SisFallUnitConverter().convert(signal)

    assert converted["mma_acc_x"].iloc[0] == pytest.approx(8.0)


def test_sisfall_converter_does_not_mutate_input():
    signal = _sample_raw_sisfall_signal()
    original = signal.copy()

    _ = SisFallUnitConverter().convert(signal)

    pd.testing.assert_frame_equal(signal, original)
