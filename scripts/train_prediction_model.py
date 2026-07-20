"""Train + evaluate a prediction-pipeline model across LOSO folds.

Usage:
    # Quick smoke run: 2 folds, few epochs -- confirms the whole
    # pipeline runs end to end on your REAL KFall data before
    # committing to a full 32-fold run.
    python scripts/train_prediction_model.py --model convlstm --max-folds 2 --max-epochs 5

    # Full LOSO run (32 folds, real training) -- this is the
    # expensive one; consider running on GPU (see below).
    python scripts/train_prediction_model.py --model convlstm
    python scripts/train_prediction_model.py --model tiny_transformer

    # Single specific fold (e.g. to debug one subject, or resume a
    # partial run by hand):
    python scripts/train_prediction_model.py --model convlstm --test-subject kfall_SA06

GPU: uses CUDA automatically if `torch.cuda.is_available()` -- if
you've only got the CPU build of torch installed (`pip show torch`
says `+cpu`), this will run on CPU, which will be considerably slower
for a full 32-fold run. Not required for a quick smoke run with
--max-folds/--max-epochs kept small.

Pipeline: trial manifest -> windows (dense, 3-class labeled) -> for
each LOSO fold: within-fold train/val subject split -> class-weighted
focal loss (weights computed from THAT fold's train split only, never
leaking the test subject's class balance) -> train with early stopping
-> evaluate on the held-out subject (per-class metrics + lead time) ->
aggregate across all folds run -> save a JSON report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

# Force line-buffered stdout so progress prints appear immediately even
# when redirected to a file (e.g. `> log.txt 2>&1 &`) -- otherwise
# Python defaults to fully block-buffered output when stdout isn't a
# real terminal, which silently delays everything until the buffer
# fills or the process exits. This caused real, reported confusion
# during the first real training run on this pipeline (looked
# indistinguishable from a hang for 30+ minutes) -- fixed here instead
# of relying on remembering to pass `python -u` every time.
sys.stdout.reconfigure(line_buffering=True)

from prediction.dataset import build_windows_manifest
from prediction.labelers import FALL, LABEL_TO_INT, NON_FALL, PRE_IMPACT
from prediction.lead_time import compute_lead_time_ms, summarize_lead_times
from prediction.loso import generate_loso_folds
from prediction.losses import FocalLoss, default_alpha_weights
from prediction.models.convlstm import ConvLSTM
from prediction.models.tiny_transformer import TinyTransformer
from prediction.training import TrainingConfig, train_one_fold
from prediction.windowing import PredictionWindowingConfig
from shared.manifest import load_manifest

REPO_ROOT = Path(__file__).parent.parent

MODEL_BUILDERS = {"convlstm": ConvLSTM, "tiny_transformer": TinyTransformer}


def _per_fold_lead_time_summary(fold_result) -> dict:
    """Compute the lead-time metric for one fold's held-out subject --
    grouped by TRIAL (a subject can have multiple fall trials), only
    over trials that actually contain a real fall (ADL trials have no
    impact frame -- `fall_impact_frame` is `None` for them, per
    `prediction.dataset.build_windows_manifest`, and lead time is
    meaningless for a trial with no fall in it)."""
    test_df = fold_result.test_windows_df.copy()
    test_df["predicted_label_id"] = fold_result.test_predicted_label_ids

    lead_times = []
    for (_, trial_id, activity_code), trial_windows in test_df.groupby(
        ["global_subject_id", "trial_id", "activity_code"], sort=False
    ):
        impact_frame = trial_windows["fall_impact_frame"].iloc[0]
        if pd.isna(impact_frame):
            continue  # ADL trial -- no impact frame, lead time not applicable

        lead_ms = compute_lead_time_ms(
            trial_windows["start_frame"].to_numpy(),
            trial_windows["predicted_label_id"].to_numpy(),
            impact_frame=int(impact_frame),
        )
        lead_times.append(lead_ms)

    if not lead_times:
        return {"n_fall_trials": 0}

    summary = summarize_lead_times(lead_times)
    return {
        "n_fall_trials": summary.n_trials,
        "n_flagged": summary.n_flagged,
        "detection_rate": summary.detection_rate,
        "mean_lead_time_ms": summary.mean_lead_time_ms,
        "median_lead_time_ms": summary.median_lead_time_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "harmonized" / "manifest.parquet"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "prediction_model"))
    parser.add_argument("--model", choices=sorted(MODEL_BUILDERS), required=True)
    parser.add_argument("--test-subject", default=None,
                         help="Run only this one LOSO fold (global_subject_id, e.g. kfall_SA06). Overrides --max-folds.")
    parser.add_argument("--max-folds", type=int, default=None,
                         help="Run only the first N folds (sorted subject order) -- for a quick smoke run, not a full 32-fold LOSO.")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--pre-impact-boost", type=float, default=2.0,
                         help="Extra focal-loss weight multiplier on the pre_impact class, on top of inverse-frequency weighting. See prediction/losses.py.")
    parser.add_argument("--window-length-s", type=float, default=1.0,
                         help="Window length in seconds. Default 1.0 matches the original blueprint spec. A shorter window (e.g. 0.5) "
                              "is worth trying since KFall's real onset-to-impact gap is only ~0.6-1.0s -- a 1.0s window can be close "
                              "to or larger than the whole event, squeezing pre_impact into a thin, hard-to-learn label band. "
                              "This flag now correctly threads into BOTH the windows manifest AND TrainingConfig.window_length_samples "
                              "(see prediction/training.py) -- previously the Dataset's window_length_samples was a disconnected "
                              "hardcoded default, which would have silently edge-padded a shorter manifest window back up to 100 "
                              "samples instead of actually shrinking it.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + ("" if device == "cuda" else "  (no CUDA GPU detected -- CPU training will be considerably slower for a full LOSO run)"))

    print(f"Loading trial manifest from {args.manifest}...")
    trial_df = load_manifest(args.manifest)

    windowing_config = PredictionWindowingConfig(window_length_s=args.window_length_s)
    window_length_samples = round(args.window_length_s * windowing_config.target_rate_hz)

    print(f"Building dense windows manifest (window_length_s={args.window_length_s}, "
          f"window_length_samples={window_length_samples}; this can take a while on the full KFall set)...")
    windows_df = build_windows_manifest(trial_df, config=windowing_config)
    print(f"  {len(windows_df)} windows across {windows_df['global_subject_id'].nunique()} subjects")
    print(f"  label distribution: {windows_df['label'].value_counts().to_dict()}")

    all_folds = generate_loso_folds(windows_df)
    if args.test_subject:
        folds = [f for f in all_folds if f.test_subject == args.test_subject]
        if not folds:
            raise ValueError(f"No subject {args.test_subject!r} found among {[f.test_subject for f in all_folds]}")
    elif args.max_folds:
        folds = all_folds[: args.max_folds]
    else:
        folds = all_folds
    print(f"Running {len(folds)} of {len(all_folds)} total LOSO folds.")

    training_config = TrainingConfig(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        device=device,
        window_length_samples=window_length_samples,
    )

    fold_reports = []
    fold_durations_s = []
    for i, fold in enumerate(folds, start=1):
        fold_start = time.monotonic()
        print(f"\n[{i}/{len(folds)}] Fold: test_subject={fold.test_subject}")

        # Class weights computed from THIS FOLD's train windows only --
        # never from the full windows_df, which would leak the held-out
        # test subject's class balance into the loss weighting.
        train_only_windows = windows_df[windows_df["global_subject_id"].isin(fold.train_subjects)]
        label_counts = train_only_windows["label"].value_counts().to_dict()
        alpha = default_alpha_weights(label_counts, pre_impact_extra_boost=args.pre_impact_boost)
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)

        model = MODEL_BUILDERS[args.model]()

        def _print_epoch_progress(epoch, train_loss, val_loss, epoch_seconds, _fold_num=i, _n_folds=len(folds)):
            print(
                f"    [{_fold_num}/{_n_folds}] epoch {epoch:>3}/{args.max_epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"({epoch_seconds:.1f}s/epoch)",
                flush=True,
            )

        result = train_one_fold(windows_df, fold, model, loss_fn, config=training_config, on_epoch_end=_print_epoch_progress)

        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{args.model}_boost{args.pre_impact_boost}_win{args.window_length_s}_{fold.test_subject}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  checkpoint saved to {checkpoint_path}")

        print(f"  best_epoch={result.best_epoch}  best_val_loss={result.best_val_loss:.4f}")
        report_text = classification_report(
            result.test_true_label_ids, result.test_predicted_label_ids,
            labels=[LABEL_TO_INT[NON_FALL], LABEL_TO_INT[PRE_IMPACT], LABEL_TO_INT[FALL]],
            target_names=[NON_FALL, PRE_IMPACT, FALL], zero_division=0,
        )
        print(report_text)

        lead_time_summary = _per_fold_lead_time_summary(result)
        print(f"  lead time: {lead_time_summary}")

        fold_duration_s = time.monotonic() - fold_start
        fold_durations_s.append(fold_duration_s)
        avg_fold_s = sum(fold_durations_s) / len(fold_durations_s)
        remaining_folds = len(folds) - i
        eta_minutes = (avg_fold_s * remaining_folds) / 60
        print(
            f"  fold took {fold_duration_s/60:.1f} min  "
            f"(avg {avg_fold_s/60:.1f} min/fold so far, "
            f"~{eta_minutes:.0f} min remaining for this run)"
        )

        fold_reports.append({
            "test_subject": fold.test_subject,
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "n_epochs_run": len(result.history.train_loss),
            "confusion_matrix": confusion_matrix(
                result.test_true_label_ids, result.test_predicted_label_ids,
                labels=[LABEL_TO_INT[NON_FALL], LABEL_TO_INT[PRE_IMPACT], LABEL_TO_INT[FALL]],
            ).tolist(),
            "classification_report_text": report_text,
            "lead_time": lead_time_summary,
        })

    # Filename encodes the config that actually varies between sweep
    # runs (boost, epochs, fold count/subject) -- a fixed filename per
    # model would silently overwrite a previous sweep's results the
    # moment a second run with different hyperparameters finishes,
    # which is exactly what nearly happened running a boost=1.0 vs.
    # boost=0.5 sweep back to back on the same --model.
    fold_tag = args.test_subject or (f"{len(folds)}folds" if args.max_folds else "allfolds")
    report_path = output_dir / f"{args.model}_boost{args.pre_impact_boost}_win{args.window_length_s}_ep{args.max_epochs}_{fold_tag}_loso_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "model": args.model,
            "n_folds_run": len(folds),
            "n_folds_total": len(all_folds),
            "training_config": vars(training_config) | {"device": device},
            "fold_reports": fold_reports,
        }, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
