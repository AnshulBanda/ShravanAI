"""XGBoost model training and evaluation for the detection pipeline.

Deliberately a thin wrapper around xgboost + sklearn's metrics -- the
real design decisions already happened upstream (windowing, features,
subject-aware splitting). This module's job is: fit on train, use val
for early stopping, evaluate on test with the metrics that actually
matter for a fall detector (recall matters more than accuracy here --
a missed fall is far costlier than a false alarm -- which is why
`scale_pos_weight` is used rather than leaving the class imbalance
unaddressed).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from detection.features import FEATURE_NAMES


@dataclass
class TrainingConfig:
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 20
    random_state: int = 42
    # If None, computed from the TRAIN split's actual class balance
    # (n_negative / n_positive) -- fall trials are a minority class in
    # both datasets (far more ADL trials than fall trials), and
    # leaving this at XGBoost's default of 1.0 would bias the model
    # toward predicting "not a fall" more than the cost of a missed
    # fall actually justifies.
    scale_pos_weight: Optional[float] = None


def _xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing {len(missing)} expected feature column(s) "
            f"(e.g. {missing[:3]}) -- did you pass a windows manifest "
            f"instead of a FEATURES dataframe from compute_features_batch?"
        )
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    return X, y


def train_model(
    train_df: pd.DataFrame, val_df: pd.DataFrame, config: Optional[TrainingConfig] = None
) -> xgb.XGBClassifier:
    """Fit an XGBoost classifier on `train_df`, using `val_df` for
    early stopping. Both must be feature DataFrames from
    `detection.features.compute_features_batch` (i.e. have all
    FEATURE_NAMES columns plus `label`).
    """
    config = config or TrainingConfig()
    X_train, y_train = _xy(train_df)
    X_val, y_val = _xy(val_df)

    scale_pos_weight = config.scale_pos_weight
    if scale_pos_weight is None:
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=config.random_state,
        eval_metric="logloss",
        early_stopping_rounds=config.early_stopping_rounds,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list  # [[tn, fp], [fn, tp]]
    n_samples: int
    n_positive: int
    classification_report_text: str


def evaluate_model(model: xgb.XGBClassifier, eval_df: pd.DataFrame) -> EvaluationResult:
    """Evaluate `model` on `eval_df` (a feature DataFrame with a real
    `label` column -- i.e. val or test, never train).
    """
    X, y = _xy(eval_df)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    # ROC-AUC is undefined with only one class present (e.g. a tiny
    # eval slice that happens to be all-ADL) -- report NaN rather than
    # letting sklearn raise, since this is a legitimate (if unlucky)
    # small-sample-size situation, not a bug.
    roc_auc = roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else float("nan")

    return EvaluationResult(
        accuracy=float(accuracy_score(y, y_pred)),
        precision=float(precision_score(y, y_pred, zero_division=0)),
        recall=float(recall_score(y, y_pred, zero_division=0)),
        f1=float(f1_score(y, y_pred, zero_division=0)),
        roc_auc=float(roc_auc),
        confusion_matrix=cm.tolist(),
        n_samples=len(y),
        n_positive=int(np.sum(y == 1)),
        classification_report_text=classification_report(
            y, y_pred, labels=[0, 1], target_names=["adl", "fall"], zero_division=0
        ),
    )


def save_model(model: xgb.XGBClassifier, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def load_model(path: Path) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return model
