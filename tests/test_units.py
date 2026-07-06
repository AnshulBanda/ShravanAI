"""Tests for shared/harmonize/units.py (Stage 3, Task 3.1)."""
import numpy as np
import pandas as pd
import pytest

from shared.harmonize.units import (
    ACCEL_COLUMNS,
    GYRO_COLUMNS,
    KFallUnitConverter,
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
        get_unit_converter("sisfall")


def test_expected_columns_defined_and_distinct():
    assert set(ACCEL_COLUMNS) == {"acc_x", "acc_y", "acc_z"}
    assert set(GYRO_COLUMNS) == {"gyro_x", "gyro_y", "gyro_z"}
    assert set(ACCEL_COLUMNS).isdisjoint(GYRO_COLUMNS)
