"""Print a trained model's window-by-window predictions for ONE real
trial, in time order, alongside the true label and the real onset/
impact frames -- a direct look at the temporal pattern, rather than
inferring it from aggregate precision/recall/lead-time numbers.

Built specifically because the boost-value sweep (2.0 / 1.0 / 0.5)
showed lead time barely moving (2774ms -> 2680ms -> 2310ms) despite
big precision/recall shifts -- suggesting the loss weighting isn't the
real lever, and the actual failure mode needs to be SEEN, not guessed
at from another aggregate number.

Usage:
    python scripts/inspect_trial_predictions.py \\
        --checkpoint results/prediction_model/checkpoints/convlstm_boost1.0_kfall_SA06.pt \\
        --model convlstm \\
        --subject kfall_SA06 --activity-code T22 --trial-id R01
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from prediction.dataset import build_windows_manifest
from prediction.features import load_augmented_window
from prediction.labelers import FALL, LABEL_TO_INT, NON_FALL, PRE_IMPACT
from prediction.models.convlstm import ConvLSTM
from prediction.models.tiny_transformer import TinyTransformer
from prediction.windowing import PredictionWindowingConfig
from shared.manifest import load_manifest

REPO_ROOT = Path(__file__).parent.parent
MODEL_BUILDERS = {"convlstm": ConvLSTM, "tiny_transformer": TinyTransformer}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "harmonized" / "manifest.parquet"))
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt file saved by scripts/train_prediction_model.py")
    parser.add_argument("--model", choices=sorted(MODEL_BUILDERS), required=True)
    parser.add_argument("--subject", required=True, help="global_subject_id, e.g. kfall_SA06")
    parser.add_argument("--activity-code", required=True, help="e.g. T22")
    parser.add_argument("--trial-id", required=True, help="e.g. R01")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trial_df = load_manifest(args.manifest)
    windows_df = build_windows_manifest(trial_df, config=PredictionWindowingConfig())

    trial_windows = windows_df[
        (windows_df["global_subject_id"] == args.subject)
        & (windows_df["activity_code"] == args.activity_code)
        & (windows_df["trial_id"] == args.trial_id)
    ].sort_values("start_frame").reset_index(drop=True)

    if len(trial_windows) == 0:
        raise ValueError(
            f"No windows found for subject={args.subject!r} "
            f"activity_code={args.activity_code!r} trial_id={args.trial_id!r}. "
            f"Check these match a real row in the manifest."
        )

    onset_frame = trial_windows["fall_onset_frame"].iloc[0]
    impact_frame = trial_windows["fall_impact_frame"].iloc[0]
    onset_frame = None if pd.isna(onset_frame) else int(onset_frame)
    impact_frame = None if pd.isna(impact_frame) else int(impact_frame)
    print(f"Trial: {args.subject} / {args.activity_code} / {args.trial_id}")
    print(f"Real onset_frame={onset_frame}  impact_frame={impact_frame}")
    print(f"{len(trial_windows)} windows, sample_rate assumed 100Hz (1 frame = 10ms)\n")

    model = MODEL_BUILDERS[args.model]()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    signal_cache: dict = {}
    windows = []
    for _, row in trial_windows.iterrows():
        windows.append(load_augmented_window(row, window_length_samples=100, signal_cache=signal_cache))
    batch = torch.from_numpy(
        np.stack(windows).transpose(0, 2, 1)  # (n_windows, n_samples, 9) -> (n_windows, 9, n_samples)
    ).float().to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        predicted_ids = probs.argmax(axis=1)

    header = f"{'start':>6} {'end':>6} {'true':<11} {'pred':<11} {'P(non_fall)':>12} {'P(pre_impact)':>14} {'P(fall)':>9}"
    print(header)
    print("-" * len(header))
    for idx, (_, row) in enumerate(trial_windows.iterrows()):
        marker = ""
        if onset_frame is not None and row["start_frame"] <= onset_frame < row["end_frame"]:
            marker = "  <-- ONSET in this window"
        if impact_frame is not None and row["start_frame"] <= impact_frame < row["end_frame"]:
            marker = "  <-- IMPACT in this window"
        true_label = row["label"]
        pred_label = INT_TO_LABEL[predicted_ids[idx]]
        flag = " " if true_label == pred_label else "*"
        print(
            f"{row['start_frame']:>6} {row['end_frame']:>6} {true_label:<11} {pred_label:<11}{flag} "
            f"{probs[idx, LABEL_TO_INT[NON_FALL]]:>11.3f} "
            f"{probs[idx, LABEL_TO_INT[PRE_IMPACT]]:>13.3f} "
            f"{probs[idx, LABEL_TO_INT[FALL]]:>8.3f}"
            f"{marker}"
        )

    print("\n* marks a window where predicted label != true label")


if __name__ == "__main__":
    main()
