"""Tests for detection/model.py."""
import numpy as np
import pandas as pd
import pytest

from detection.features import FEATURE_NAMES
from detection.model import (
    TrainingConfig,
    _xy,
    evaluate_model,
    load_model,
    save_model,
    train_model,
)


def _synthetic_features_df(n_per_class: int = 60, random_state: int = 0) -> pd.DataFrame:
    """Features DataFrame with all FEATURE_NAMES columns filled with
    noise, EXCEPT one feature ('jerk_max_abs') that's strongly
    correlated with the label -- gives a trained model something real
    to learn, so tests can check it actually learned it rather than
    just running without crashing.
    """
    rng = np.random.default_rng(random_state)
    n = n_per_class * 2
    labels = np.array([0] * n_per_class + [1] * n_per_class)

    data = {name: rng.normal(0, 1, n) for name in FEATURE_NAMES}
    # Falls (label=1) have a much larger jerk_max_abs than ADLs.
    data["jerk_max_abs"] = np.where(labels == 1, rng.normal(20, 2, n), rng.normal(1, 0.5, n))

    df = pd.DataFrame(data)
    df["label"] = labels
    df["global_subject_id"] = [f"kfall_S{i % 10:02d}" for i in range(n)]
    df["dataset"] = "kfall"
    return df


def test_xy_extracts_correct_shapes():
    df = _synthetic_features_df(n_per_class=10)
    X, y = _xy(df)

    assert X.shape == (20, len(FEATURE_NAMES))
    assert y.shape == (20,)
    assert set(y) == {0, 1}


def test_xy_raises_clear_error_on_missing_feature_columns():
    df = pd.DataFrame({"label": [0, 1], "some_other_column": [1.0, 2.0]})

    with pytest.raises(ValueError, match="missing"):
        _xy(df)


def test_train_model_learns_the_separable_feature():
    df = _synthetic_features_df(n_per_class=100, random_state=1)
    train_df = df.iloc[:160]
    val_df = df.iloc[160:]

    model = train_model(train_df, val_df, TrainingConfig(n_estimators=50))
    result = evaluate_model(model, val_df)

    # With one cleanly separable feature and 100+ samples/class, the
    # model should do far better than chance -- not necessarily
    # perfect (noise on the other 53 features), but clearly learned.
    assert result.accuracy > 0.85
    assert result.recall > 0.8


def test_evaluate_model_confusion_matrix_shape_and_totals():
    df = _synthetic_features_df(n_per_class=50, random_state=2)
    train_df, val_df = df.iloc[:80], df.iloc[80:]

    model = train_model(train_df, val_df, TrainingConfig(n_estimators=30))
    result = evaluate_model(model, val_df)

    assert len(result.confusion_matrix) == 2
    assert len(result.confusion_matrix[0]) == 2
    total = sum(sum(row) for row in result.confusion_matrix)
    assert total == result.n_samples == len(val_df)


def test_evaluate_model_handles_single_class_eval_set_without_crashing():
    df = _synthetic_features_df(n_per_class=50, random_state=3)
    train_df, val_df = df.iloc[:80], df.iloc[80:]
    model = train_model(train_df, val_df, TrainingConfig(n_estimators=30))

    single_class_df = df[df["label"] == 0].iloc[:10]
    result = evaluate_model(model, single_class_df)

    assert np.isnan(result.roc_auc)  # undefined with only one class present
    assert result.n_samples == 10


def test_save_and_load_model_round_trip_produces_identical_predictions(tmp_path):
    df = _synthetic_features_df(n_per_class=60, random_state=4)
    train_df, val_df = df.iloc[:100], df.iloc[100:]
    model = train_model(train_df, val_df, TrainingConfig(n_estimators=30))

    model_path = tmp_path / "model.json"
    save_model(model, model_path)
    loaded_model = load_model(model_path)

    X, _ = _xy(val_df)
    original_proba = model.predict_proba(X)[:, 1]
    loaded_proba = loaded_model.predict_proba(X)[:, 1]

    np.testing.assert_allclose(original_proba, loaded_proba, rtol=1e-5)


def test_save_model_creates_parent_directories(tmp_path):
    df = _synthetic_features_df(n_per_class=30, random_state=5)
    train_df, val_df = df.iloc[:40], df.iloc[40:]
    model = train_model(train_df, val_df, TrainingConfig(n_estimators=20))

    nested_path = tmp_path / "nested" / "dir" / "model.json"
    save_model(model, nested_path)

    assert nested_path.exists()
