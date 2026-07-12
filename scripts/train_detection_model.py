"""Train + evaluate the detection XGBoost model, end to end.

Usage:
    python scripts/train_detection_model.py
    python scripts/train_detection_model.py --manifest path/to/manifest.parquet
    python scripts/train_detection_model.py --datasets kfall sisfall

Pipeline: trial manifest -> windows -> handcrafted features (cached to
disk so re-runs don't recompute) -> subject-aware train/val/test split
-> train XGBoost with early stopping -> evaluate on held-out test ->
save model + a JSON report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from detection.dataset import build_windows_manifest
from detection.features import compute_features_batch
from detection.model import TrainingConfig, evaluate_model, save_model, train_model
from detection.split import SplitConfig, split_by_subject
from detection.windowing import WindowingConfig
from shared.manifest import load_manifest

REPO_ROOT = Path(__file__).parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default=str(REPO_ROOT / "data" / "harmonized" / "manifest.parquet"),
        help="Path to the trial-level manifest (from harmonize_dataset.py).",
    )
    parser.add_argument(
        "--output-dir", default=str(REPO_ROOT / "results" / "detection_model"),
        help="Where to write the trained model, features cache, and evaluation report.",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Restrict to these datasets (default: all datasets in the manifest).",
    )
    parser.add_argument(
        "--force-recompute-features", action="store_true",
        help="Recompute features even if a cache file already exists.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading trial manifest from {args.manifest}...")
    trial_df = load_manifest(args.manifest)
    print(f"  {len(trial_df)} trials, datasets: {sorted(trial_df['dataset'].unique())}")

    windowing_config = WindowingConfig()
    print(f"Building windows manifest (window={windowing_config.window_length_s}s, "
          f"stride={windowing_config.stride_s}s)...")
    windows_df = build_windows_manifest(trial_df, config=windowing_config, datasets=args.datasets)
    print(f"  {len(windows_df)} windows across "
          f"{windows_df['global_subject_id'].nunique()} subjects "
          f"({int((windows_df['label'] == 1).sum())} fall-labeled, "
          f"{int((windows_df['label'] == 0).sum())} adl-labeled)")

    features_cache_path = output_dir / "features_cache.parquet"
    if features_cache_path.exists() and not args.force_recompute_features:
        print(f"Loading cached features from {features_cache_path}...")
        features_df = pd.read_parquet(features_cache_path)
    else:
        print("Computing handcrafted features (this can take a while on the full dataset)...")
        features_df = compute_features_batch(windows_df, windowing_config=windowing_config)
        features_df.to_parquet(features_cache_path, index=False)
        print(f"  Features cached to {features_cache_path}")

    print("Splitting train/val/test (subject-aware, per-dataset stratified)...")
    train_df, val_df, test_df = split_by_subject(features_df, SplitConfig())
    print(f"  train={len(train_df)} ({train_df['global_subject_id'].nunique()} subjects)")
    print(f"  val=  {len(val_df)} ({val_df['global_subject_id'].nunique()} subjects)")
    print(f"  test= {len(test_df)} ({test_df['global_subject_id'].nunique()} subjects)")

    print("Training XGBoost model (early stopping on val)...")
    model = train_model(train_df, val_df, TrainingConfig())

    print("Evaluating on held-out TEST set...")
    result = evaluate_model(model, test_df)
    print(f"  Accuracy:  {result.accuracy:.4f}")
    print(f"  Precision: {result.precision:.4f}")
    print(f"  Recall:    {result.recall:.4f}  <- most important for a fall detector")
    print(f"  F1:        {result.f1:.4f}")
    print(f"  ROC-AUC:   {result.roc_auc:.4f}")
    print(f"  Confusion matrix [[TN,FP],[FN,TP]]: {result.confusion_matrix}")
    print()
    print(result.classification_report_text)

    model_path = output_dir / "xgboost_model.json"
    save_model(model, model_path)
    print(f"Model saved to {model_path}")

    report_path = output_dir / "evaluation_report.json"
    report = {
        "datasets": args.datasets or sorted(trial_df["dataset"].unique().tolist()),
        "n_trials": len(trial_df),
        "n_windows": len(windows_df),
        "train_size": len(train_df), "val_size": len(val_df), "test_size": len(test_df),
        "train_subjects": train_df["global_subject_id"].nunique(),
        "val_subjects": val_df["global_subject_id"].nunique(),
        "test_subjects": test_df["global_subject_id"].nunique(),
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "roc_auc": result.roc_auc,
        "confusion_matrix": result.confusion_matrix,
        "n_test_samples": result.n_samples,
        "n_test_positive": result.n_positive,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
