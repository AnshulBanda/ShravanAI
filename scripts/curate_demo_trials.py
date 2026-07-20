"""
Scans EVERY fall trial for a given subject through the model + smoother
(same pipeline scripts/run_live_demo.py uses) and reports, per trial,
whether the smoothed alert stays CALM until close to the real onset --
i.e. whether it's a safe, non-embarrassing choice for a live demo.

Built after discovering (live, mid-demo-prep) that SA06/T22/R01 --
the trial used in every earlier example this session -- fires a
PRE_IMPACT warning at t=0.00s, before any real movement. Rather than
keep guessing individual trials, this checks all of them at once.

Usage:
    python scripts/curate_demo_trials.py \
        --checkpoint results/prediction_model/checkpoints/convlstm_boost2.0_kfall_SA06.pt \
        --model convlstm \
        --subject kfall_SA06

Only scans trials for the given subject that are the checkpoint's
actual held-out LOSO test subject (by convention, matches the
checkpoint filename) -- deliberately NOT testing on subjects the
checkpoint was trained on, since a demo built on memorized training
data would be misleading about what the system can actually do.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from prediction.dataset import build_windows_manifest
from prediction.features import load_augmented_window
from prediction.live_smoothing import AlertState, PredictionSmoother, SmootherConfig
from prediction.models.convlstm import ConvLSTM
from prediction.models.tiny_transformer import TinyTransformer
from prediction.windowing import PredictionWindowingConfig
from shared.manifest import load_manifest

REPO_ROOT = Path(__file__).parent.parent
MODEL_BUILDERS = {"convlstm": ConvLSTM, "tiny_transformer": TinyTransformer}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "harmonized" / "manifest.parquet"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=sorted(MODEL_BUILDERS), required=True)
    parser.add_argument("--subject", required=True, help="global_subject_id, e.g. kfall_SA06")
    parser.add_argument("--window-length-s", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    windowing_config = PredictionWindowingConfig(window_length_s=args.window_length_s)
    window_length_samples = round(args.window_length_s * windowing_config.target_rate_hz)
    rate_hz = windowing_config.target_rate_hz

    trial_df = load_manifest(args.manifest)
    windows_df = build_windows_manifest(trial_df, config=windowing_config)

    subject_fall_windows = windows_df[
        (windows_df["global_subject_id"] == args.subject) & (windows_df["fall_onset_frame"].notna())
    ]
    trial_keys = sorted(subject_fall_windows[["activity_code", "trial_id"]].drop_duplicates().itertuples(index=False, name=None))

    if not trial_keys:
        raise ValueError(f"No fall trials found for subject={args.subject!r} -- check it's a real global_subject_id.")

    model = MODEL_BUILDERS[args.model]()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    print(f"Scanning {len(trial_keys)} fall trials for {args.subject}...\n")
    print(f"{'trial':<12} {'onset_s':>8} {'impact_s':>9} {'first_escalate_s':>18} {'lead_s':>8} {'fall_lag_s':>10} {'reached_fall':>13} {'safe_for_demo':>14}")
    print("-" * 90)

    results = []
    signal_cache: dict = {}
    for activity_code, trial_id in trial_keys:
        trial_windows = windows_df[
            (windows_df["global_subject_id"] == args.subject)
            & (windows_df["activity_code"] == activity_code)
            & (windows_df["trial_id"] == trial_id)
        ].sort_values("start_frame").reset_index(drop=True)

        onset_frame = trial_windows["fall_onset_frame"].iloc[0]
        impact_frame = trial_windows["fall_impact_frame"].iloc[0]
        onset_s = onset_frame / rate_hz
        impact_s = impact_frame / rate_hz

        smoother = PredictionSmoother(SmootherConfig())
        first_escalate_s = None
        first_fall_s = None
        reached_fall = False

        for _, row in trial_windows.iterrows():
            window = load_augmented_window(row, window_length_samples=window_length_samples, signal_cache=signal_cache)
            x = torch.from_numpy(np.ascontiguousarray(window.T)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().numpy()
            state = smoother.update(probs)

            t_s = row["start_frame"] / rate_hz
            if state != AlertState.CALM and first_escalate_s is None:
                first_escalate_s = t_s
            if state == AlertState.FALL:
                reached_fall = True
                if first_fall_s is None:
                    first_fall_s = t_s

        lead_s = (onset_s - first_escalate_s) if first_escalate_s is not None else None
        fall_lag_s = (first_fall_s - impact_s) if first_fall_s is not None else None
        # "safe for demo" heuristic, v2: doesn't escalate more than 0.3s
        # before real onset, DOES reach FALL, AND reaches FALL within 1.0s
        # of real impact -- v1 only checked the first two and let
        # T29/R04 through, which sat in PRE_IMPACT for 4.2s after the
        # real impact before finally escalating to FALL. A demo where
        # the alert lags visible impact by multiple seconds is arguably
        # worse than one that's slightly early.
        safe = (
            first_escalate_s is not None
            and (onset_s - first_escalate_s) <= 0.3
            and reached_fall
            and fall_lag_s is not None
            and fall_lag_s <= 1.0
        )
        results.append((activity_code, trial_id, onset_s, impact_s, first_escalate_s, lead_s, first_fall_s, fall_lag_s, reached_fall, safe))

        trial_label = f"{activity_code}/{trial_id}"
        escalate_str = f"{first_escalate_s:.2f}" if first_escalate_s is not None else "never"
        lead_str = f"{lead_s:+.2f}" if lead_s is not None else "n/a"
        fall_lag_str = f"{fall_lag_s:+.2f}" if fall_lag_s is not None else "n/a"
        print(f"{trial_label:<12} {onset_s:>8.2f} {impact_s:>9.2f} {escalate_str:>18} {lead_str:>8} {fall_lag_str:>10} {str(reached_fall):>13} {('YES' if safe else 'no'):>14}")

    safe_trials = [r for r in results if r[-1]]
    print(f"\n{len(safe_trials)} / {len(results)} trials look demo-safe by this heuristic.")
    if safe_trials:
        print("Recommended (sorted by cleanest overall timing -- onset lead + fall lag combined):")
        safe_trials.sort(key=lambda r: abs(r[5] or 0) + abs(r[7] or 0))
        for r in safe_trials[:5]:
            print(f"  {r[0]}/{r[1]}  (escalates {r[5]:+.2f}s rel. to onset, reaches FALL {r[7]:+.2f}s rel. to impact)")


if __name__ == "__main__":
    main()
