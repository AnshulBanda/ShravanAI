"""
Aggregates prediction errors across an ENTIRE held-out LOSO fold's test
windows, binned by each window's signed frame-distance to the trial's
real onset frame. Built after three hand-picked trials (SA06/T22,
SA06/T30, SA18/T22) each showed a DIFFERENT failure shape (early
false-positive, early false-positive, late/missed) -- single-trial
inspection stopped being informative, so this looks at the whole
distribution instead of one trial at a time.

Only ADL trials get skipped from the distance-to-onset binning (they
have no onset_frame at all); fall trials contribute one row per window.

Usage:
    python scripts/analyze_temporal_errors.py \
        --checkpoint results/prediction_model/checkpoints/convlstm_boost2.0_kfall_SA06.pt \
        --model convlstm \
        --subject kfall_SA06
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from prediction.dataset import build_windows_manifest
from prediction.labelers import LABEL_TO_INT
from prediction.loso import LOSOFold, get_fold_masks
from prediction.models.convlstm import ConvLSTM
from prediction.models.tiny_transformer import TinyTransformer
from prediction.torch_dataset import PredictionWindowDataset
from prediction.windowing import PredictionWindowingConfig
from shared.manifest import load_manifest

REPO_ROOT = Path(__file__).parent.parent
MODEL_BUILDERS = {"convlstm": ConvLSTM, "tiny_transformer": TinyTransformer}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

# Bin edges in FRAMES relative to onset_frame (window's start_frame - onset_frame).
# Negative = before onset, positive = at/after onset. 10-frame (100ms) bins
# near the boundary, coarser further out where less precision is needed.
BIN_EDGES = [-400, -300, -200, -100, -50, -20, 0, 20, 50, 100, 200, 300, 400]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "harmonized" / "manifest.parquet"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=sorted(MODEL_BUILDERS), required=True)
    parser.add_argument("--subject", required=True, help="global_subject_id, e.g. kfall_SA06 -- must match the checkpoint's held-out test subject")
    parser.add_argument("--window-length-s", type=float, default=1.0,
                         help="MUST match the window_length_s the checkpoint was trained with (see the checkpoint filename's win{X} segment).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    windowing_config = PredictionWindowingConfig(window_length_s=args.window_length_s)
    window_length_samples = round(args.window_length_s * windowing_config.target_rate_hz)

    trial_df = load_manifest(args.manifest)
    windows_df = build_windows_manifest(trial_df, config=windowing_config)

    fold = LOSOFold(
        test_subject=args.subject,
        train_subjects=tuple(sorted(set(windows_df["global_subject_id"]) - {args.subject})),
    )
    _, test_mask = get_fold_masks(windows_df, fold)
    test_df = windows_df[test_mask].reset_index(drop=True)
    if test_df.empty:
        raise ValueError(f"No test windows found for subject={args.subject!r} -- check the value matches a real global_subject_id.")

    model = MODEL_BUILDERS[args.model]()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    loader = DataLoader(
        PredictionWindowDataset(test_df, window_length_samples=window_length_samples),
        batch_size=256, shuffle=False,
    )
    predicted_ids = []
    with torch.no_grad():
        for x, _y in loader:
            logits = model(x.to(device))
            predicted_ids.append(logits.argmax(dim=1).cpu().numpy())
    test_df = test_df.copy()
    test_df["predicted_label_id"] = np.concatenate(predicted_ids)
    test_df["predicted_label"] = test_df["predicted_label_id"].map(INT_TO_LABEL)
    test_df["correct"] = test_df["predicted_label"] == test_df["label"]

    # --- Fall-trial windows, binned by distance to onset ---
    fall_windows = test_df[test_df["fall_onset_frame"].notna()].copy()
    fall_windows["dist_to_onset"] = fall_windows["start_frame"] - fall_windows["fall_onset_frame"]
    fall_windows["false_pre_impact"] = (
        (fall_windows["predicted_label"] == "pre_impact") & (fall_windows["label"] != "pre_impact")
    )

    fall_windows["bin"] = pd.cut(fall_windows["dist_to_onset"], bins=BIN_EDGES)

    print(f"=== Fall-trial windows binned by distance to onset (frames), subject={args.subject} ===")
    print(f"{'bin (frames from onset)':<28} {'n':>6} {'accuracy':>10} {'false_pre_impact_rate':>22}")
    print("-" * 70)
    for bin_label, group in fall_windows.groupby("bin", observed=True):
        if len(group) == 0:
            continue
        acc = group["correct"].mean()
        fp_rate = group["false_pre_impact"].mean()
        print(f"{str(bin_label):<28} {len(group):>6} {acc:>10.3f} {fp_rate:>22.3f}")

    # --- ADL-trial windows (no onset at all) -- sanity check, should be low false-positive ---
    adl_windows = test_df[test_df["fall_onset_frame"].isna()]
    if len(adl_windows) > 0:
        adl_fp_rate = (adl_windows["predicted_label"] == "pre_impact").mean()
        print(f"\n=== ADL-trial windows (no onset ever), subject={args.subject} ===")
        print(f"n={len(adl_windows)}  overall_accuracy={adl_windows['correct'].mean():.3f}  "
              f"pre_impact_false_positive_rate={adl_fp_rate:.3f}")


if __name__ == "__main__":
    main()
