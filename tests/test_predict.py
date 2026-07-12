"""Tests for detection/predict.py."""
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from detection.dataset import CHANNELS
from detection.features import FEATURE_NAMES
from detection.model import TrainingConfig, save_model, train_model
from detection.predict import predict_from_manifest, predict_single_window
from detection.windowing import WindowingConfig
from shared.manifest import ManifestRow, write_manifest


def _train_toy_model(tmp_path):
    """Trains a tiny real model on synthetic features (same approach
    as test_model.py) and saves it, for predict.py's tests to load.
    """
    rng = np.random.default_rng(0)
    n_per_class = 60
    n = n_per_class * 2
    labels = np.array([0] * n_per_class + [1] * n_per_class)
    data = {name: rng.normal(0, 1, n) for name in FEATURE_NAMES}
    data["jerk_max_abs"] = np.where(labels == 1, rng.normal(20, 2, n), rng.normal(1, 0.5, n))
    df = pd.DataFrame(data)
    df["label"] = labels

    model = train_model(df.iloc[:100], df.iloc[100:], TrainingConfig(n_estimators=30))
    model_path = tmp_path / "model.json"
    save_model(model, model_path)
    return model_path


def _write_trial_parquet(path, n_rows, jerky: bool):
    """A harmonized-format parquet file. `jerky=True` produces a sharp
    spike (fall-like); `jerky=False` stays quiet (ADL-like) -- so the
    real feature pipeline downstream produces a jerk_max_abs value in
    the same range the toy model above was trained to associate with
    each class.
    """
    rng = np.random.default_rng(1)
    acc = rng.normal(0, 0.05, (n_rows, 3)).astype(np.float32)
    acc[:, 2] += 1.0  # resting gravity
    if jerky and n_rows > 20:
        acc[n_rows // 2, 0] += 8.0  # sharp spike
    gyro = rng.normal(0, 2.0, (n_rows, 3)).astype(np.float32)

    df = pd.DataFrame({
        "time_s": np.arange(n_rows) / 100.0,
        "acc_x": acc[:, 0], "acc_y": acc[:, 1], "acc_z": acc[:, 2],
        "gyro_x": gyro[:, 0], "gyro_y": gyro[:, 1], "gyro_z": gyro[:, 2],
    })
    df.to_parquet(path, index=False)
    return len(df)


def _trial_row(**overrides) -> dict:
    defaults = dict(
        dataset="kfall", subject_id="SA06", activity_code="T01", trial_id="R01",
        label="adl", duration_s=2.0, sample_rate_hz=100.0, accepted=True,
        calibration_source="T01", harmonized_path="/fake.parquet",
        fall_onset_frame=None, fall_impact_frame=None,
    )
    defaults.update(overrides)
    return asdict(ManifestRow(**defaults))


def test_predict_from_manifest_returns_expected_columns(tmp_path):
    model_path = _train_toy_model(tmp_path)
    trial_path = tmp_path / "trial.parquet"
    n_rows = _write_trial_parquet(trial_path, n_rows=200, jerky=False)

    manifest_path = tmp_path / "manifest.parquet"
    write_manifest(
        [ManifestRow(**_trial_row(
            duration_s=n_rows / 100.0, harmonized_path=str(trial_path), label="adl",
        ))],
        manifest_path,
    )

    result = predict_from_manifest(model_path, manifest_path, windowing_config=WindowingConfig())

    assert len(result) > 0
    for col in ["dataset", "global_subject_id", "predicted_label", "fall_probability"]:
        assert col in result.columns
    assert result["fall_probability"].between(0, 1).all()


def test_predict_from_manifest_empty_manifest_returns_empty_with_columns(tmp_path):
    model_path = _train_toy_model(tmp_path)
    manifest_path = tmp_path / "manifest.parquet"
    write_manifest(
        [ManifestRow(**_trial_row(accepted=False))],  # excluded by query_detection_trials
        manifest_path,
    )

    result = predict_from_manifest(model_path, manifest_path, windowing_config=WindowingConfig())

    assert len(result) == 0
    assert "predicted_label" in result.columns


def test_predict_single_window_returns_probability_and_label(tmp_path):
    model_path = _train_toy_model(tmp_path)
    window = np.zeros((200, len(CHANNELS)), dtype=np.float32)
    window[:, 2] = 1.0  # quiet, resting

    result = predict_single_window(model_path, window)

    assert "fall_probability" in result
    assert "predicted_label" in result
    assert 0.0 <= result["fall_probability"] <= 1.0
    assert result["predicted_label"] in (0, 1)


def test_predict_single_window_jerky_window_scores_higher_than_quiet(tmp_path):
    # Not a guarantee for every possible model/seed, but with the toy
    # model's clear jerk_max_abs signal, a genuinely spiky window
    # should score noticeably higher fall-probability than a quiet one.
    model_path = _train_toy_model(tmp_path)

    quiet = np.zeros((200, len(CHANNELS)), dtype=np.float32)
    quiet[:, 2] = 1.0

    spiky = quiet.copy()
    spiky[100, 0] = 20.0  # single large spike

    quiet_result = predict_single_window(model_path, quiet)
    spiky_result = predict_single_window(model_path, spiky)

    assert spiky_result["fall_probability"] > quiet_result["fall_probability"]
