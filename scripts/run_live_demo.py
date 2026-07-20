"""
Live demo: replays a REAL trial's IMU windows through the model, one
window at a time in chronological order (like a live stream would),
running each raw prediction through PredictionSmoother before printing
it -- so what's displayed is the stable, demo-safe alert state, not the
raw per-window flicker.

This replays REAL, already-recorded sensor data -- it is not simulated
or fabricated. What makes it "live" is the pacing (windows are revealed
one at a time, at real-time cadence by default) and that the model has
never seen this specific trial if it's the checkpoint's held-out test
subject.

Usage (matches inspect_trial_predictions.py's argument style):
    python scripts/run_live_demo.py \
        --checkpoint results/prediction_model/checkpoints/convlstm_boost2.0_kfall_SA06.pt \
        --model convlstm \
        --subject kfall_SA06 --activity-code T22 --trial-id R01

Add --fast to skip the real-time pacing (instant replay, useful for a
dry run) or --speed 2.0 to replay at 2x real-time.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import numpy as np

from prediction.dataset import build_windows_manifest
from prediction.features import load_augmented_window
from prediction.live_smoothing import AlertState, PredictionSmoother, SmootherConfig
from prediction.models.convlstm import ConvLSTM
from prediction.models.tiny_transformer import TinyTransformer
from prediction.windowing import PredictionWindowingConfig
from shared.manifest import load_manifest

REPO_ROOT = Path(__file__).parent.parent
MODEL_BUILDERS = {"convlstm": ConvLSTM, "tiny_transformer": TinyTransformer}

# ANSI colors -- CALM is intentionally quiet/dim, PRE_IMPACT and FALL are
# loud, since the whole point of the display is that an escalation should
# be impossible to miss when watching live.
_COLOR = {
    AlertState.CALM: "\033[2m",       # dim
    AlertState.PRE_IMPACT: "\033[33m",  # yellow
    AlertState.FALL: "\033[91m",        # bright red
}
_RESET = "\033[0m"

_BANNER = {
    AlertState.CALM: "  calm",
    AlertState.PRE_IMPACT: "  \u26a0  PRE-IMPACT WARNING",
    AlertState.FALL: "  \U0001f6a8 FALL DETECTED \U0001f6a8",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "harmonized" / "manifest.parquet"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=sorted(MODEL_BUILDERS), required=True)
    parser.add_argument("--subject", required=True, help="global_subject_id, e.g. kfall_SA06")
    parser.add_argument("--activity-code", required=True, help="e.g. T22")
    parser.add_argument("--trial-id", required=True, help="e.g. R01")
    parser.add_argument("--window-length-s", type=float, default=1.0,
                         help="MUST match the checkpoint's training window_length_s (see its win{X} filename segment).")
    parser.add_argument("--fast", action="store_true", help="Skip real-time pacing, replay instantly.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (2.0 = 2x real-time). Ignored if --fast.")
    parser.add_argument("--show-ground-truth", action="store_true",
                         help="Also print the real onset/impact frames for narration purposes. "
                              "Honest framing for a demo: a deployed system would NOT have this -- "
                              "it's shown here so you can narrate 'the real onset was here, the model "
                              "flagged it here' while presenting.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    windowing_config = PredictionWindowingConfig(window_length_s=args.window_length_s)
    window_length_samples = round(args.window_length_s * windowing_config.target_rate_hz)
    stride_s = windowing_config.stride_s

    trial_df = load_manifest(args.manifest)
    windows_df = build_windows_manifest(trial_df, config=windowing_config)

    trial_windows = windows_df[
        (windows_df["global_subject_id"] == args.subject)
        & (windows_df["activity_code"] == args.activity_code)
        & (windows_df["trial_id"] == args.trial_id)
    ].sort_values("start_frame").reset_index(drop=True)

    if trial_windows.empty:
        raise ValueError(
            f"No windows found for subject={args.subject!r} activity_code={args.activity_code!r} "
            f"trial_id={args.trial_id!r}. Check these match a real row in the manifest."
        )

    onset_frame = trial_windows["fall_onset_frame"].iloc[0]
    impact_frame = trial_windows["fall_impact_frame"].iloc[0]

    model = MODEL_BUILDERS[args.model]()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    smoother = PredictionSmoother(SmootherConfig())

    print(f"\n{'='*60}")
    print(f"  ShravanAI live demo -- {args.subject} / {args.activity_code} / {args.trial_id}")
    print(f"{'='*60}")
    if args.show_ground_truth and onset_frame == onset_frame:  # NaN check
        print(f"  (narration only -- not shown to a real deployed system)")
        print(f"  real onset_frame={int(onset_frame)}  impact_frame={int(impact_frame)}")
    print(f"{'='*60}\n")

    prev_state = None
    signal_cache: dict = {}
    for _, row in trial_windows.iterrows():
        window = load_augmented_window(row, window_length_samples=window_length_samples, signal_cache=signal_cache)
        # augmented is (n_samples, 9) -- transpose to channel-first (9,
        # n_samples) before adding the batch dim, matching exactly what
        # PredictionWindowDataset.__getitem__ does (see prediction/torch_dataset.py).
        x = torch.from_numpy(np.ascontiguousarray(window.T)).float().unsqueeze(0).to(device)  # (1, 9, window_length_samples)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().numpy()

        state = smoother.update(probs)
        color = _COLOR[state]
        t_s = row["start_frame"] / windowing_config.target_rate_hz

        marker = ""
        if state != prev_state:
            marker = f"{color}{_BANNER[state]}{_RESET}"
        line = f"  t={t_s:5.2f}s  P(non_fall/pre_impact/fall)=[{probs[0]:.2f} {probs[1]:.2f} {probs[2]:.2f}]  {color}{state.value:<12}{_RESET} {marker}"
        print(line)
        prev_state = state

        if not args.fast:
            time.sleep(stride_s / args.speed)

    print(f"\n{'='*60}")
    print(f"  Replay finished. Final state: {prev_state.value if prev_state else 'n/a'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
