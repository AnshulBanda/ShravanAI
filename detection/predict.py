"""Inference utilities: apply a trained detection model to get
predictions on new or held-out data.

Two entry points:
- `predict_from_manifest`: batch predictions over every window in a
  trial manifest (e.g. your held-out test set, or a brand new dataset
  run through harmonization later) -- this is the "get predictions on
  test data" workflow.
- `predict_single_window`: predict on one raw (n_samples, 6) array
  directly, for ad hoc / single-sample use outside the manifest flow.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from detection.dataset import build_windows_manifest, load_window
from detection.features import FEATURE_NAMES, compute_window_features
from detection.model import load_model
from detection.windowing import WindowingConfig
from shared.manifest import load_manifest


def predict_from_manifest(
    model_path: Path,
    trial_manifest_path: Path,
    windowing_config: Optional[WindowingConfig] = None,
    datasets: Optional[list[str]] = None,
) -> pd.DataFrame:
    """End-to-end: trial manifest -> windows -> features -> predictions.

    Returns one row per window with identifying columns (dataset,
    global_subject_id, activity_code, trial_id, window_index), the
    TRUE label (present whenever the source manifest has one -- useful
    for evaluating on a labeled held-out set), predicted label, and
    fall probability.
    """
    windowing_config = windowing_config or WindowingConfig()
    model = load_model(model_path)

    trial_df = load_manifest(trial_manifest_path)
    windows_df = build_windows_manifest(trial_df, config=windowing_config, datasets=datasets)

    if len(windows_df) == 0:
        return windows_df.assign(predicted_label=[], fall_probability=[])

    cache: dict = {}
    feature_rows = []
    for _, window_row in windows_df.iterrows():
        window = load_window(
            window_row, windowing_config.window_length_samples, signal_cache=cache
        )
        feature_rows.append(
            compute_window_features(window, sample_rate_hz=windowing_config.target_rate_hz)
        )

    features_df = pd.DataFrame(feature_rows)
    X = features_df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    proba = model.predict_proba(X)[:, 1]

    result = windows_df[
        ["dataset", "global_subject_id", "activity_code", "trial_id", "window_index", "label"]
    ].copy()
    result["predicted_label"] = (proba >= 0.5).astype(int)
    result["fall_probability"] = proba
    return result


def predict_single_window(
    model_path: Path, window: np.ndarray, sample_rate_hz: float = 100.0
) -> dict:
    """Predict on one raw window array (shape (n_samples, 6), CHANNELS
    order: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z).

    Loads the model fresh on every call -- fine for occasional/ad hoc
    use, wasteful if called in a tight loop over many windows (use
    `predict_from_manifest` for batches, which loads the model once).
    """
    model = load_model(model_path)
    features = compute_window_features(window, sample_rate_hz=sample_rate_hz)
    X = np.array([[features[name] for name in FEATURE_NAMES]], dtype=np.float32)
    proba = float(model.predict_proba(X)[0, 1])
    return {"fall_probability": proba, "predicted_label": int(proba >= 0.5)}
