"""Stage 3, Task 3.11 -- visual QA pass (human-in-the-loop, NOT automated).

This is a one-off exploratory script, not imported by anything and not
part of the test suite. Its entire purpose is to let a human eyeball
harmonization quality on REAL KFall data before calling Stage 3 done.
It intentionally does not assert/fail on anything -- read the printed
output and the saved plots yourself.

Usage:
    python notebooks/stage3_visual_qa.py
    python notebooks/stage3_visual_qa.py --subjects SA06 SA07 SA08
    python notebooks/stage3_visual_qa.py --max-trials-per-subject 3

Requires real KFall data under data/raw/kfall/{sensor_data,label_data}
(see configs/datasets/kfall.yaml). As of this checkpoint, only SA06 is
available locally -- this script is written to degrade gracefully to
whatever subjects are actually present rather than assuming a fixed
roster, so re-running it later with more subjects downloaded just
produces more coverage without any code changes.

What this produces (per the Task 3.11 spec):
1. Raw-vs-harmonized signal overlay plots for a handful of real trials,
   spread across whichever subjects/activity types are available.
2. A histogram/table of calibration source across all real KFall
   subjects downloaded so far (should be mostly "T01").
3. For every real fall trial harmonized cleanly (not quarantined), the
   labeled onset/impact frames vs. the frame of peak harmonized-signal
   magnitude near impact -- extending the manual SA06 T22 check
   (frame 202 vs. labeled impact 208) to whatever other fall trials are
   available.

All outputs are written under results/stage3_visual_qa/ so nothing here
touches data/harmonized/ (that's Task 3.10's job, already done).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless -- this script only writes PNGs, never shows()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared.config import load_config
from shared.harmonize.axis_alignment import (
    STANDING_INITIATED_TASK_IDS,
    calibrate_subject,
    resolve_group_fallback,
    summarize_calibration_sources,
)
from shared.harmonize.pipeline import HarmonizationConfig, harmonize_trial
from shared.harmonize.units import ACCEL_COLUMNS, GYRO_COLUMNS
from shared.harmonize.validation import validate_harmonized_trial
from shared.io.readers_kfall import load_all_trials

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "stage3_visual_qa"


def _pick_qa_trials(trials, subjects_filter, max_per_subject):
    """Pick a handful of trials per subject: at least one ADL, one fall."""
    by_subject: dict[str, list] = {}
    for t in trials:
        if subjects_filter and t.metadata.subject_id not in subjects_filter:
            continue
        by_subject.setdefault(t.metadata.subject_id, []).append(t)

    selected = []
    for subject_id, subject_trials in sorted(by_subject.items()):
        adl = [t for t in subject_trials if t.metadata.label == "adl"]
        falls = [t for t in subject_trials if t.metadata.label == "fall"]
        picks = (adl[:1] + falls[:1])
        # Top up to max_per_subject with whatever's left, preferring variety.
        # NOTE: compare by identity, not `in`/`==` -- ParsedTrial holds a
        # DataFrame field, so `==` raises ("truth value of a DataFrame is
        # ambiguous") rather than doing the identity check we want here.
        picked_ids = {id(t) for t in picks}
        remaining = [t for t in subject_trials if id(t) not in picked_ids]
        picks += remaining[: max(0, max_per_subject - len(picks))]
        selected.extend(picks[:max_per_subject])
    return selected, by_subject


def _plot_raw_vs_harmonized(trial, harmonized: pd.DataFrame, out_path: Path) -> None:
    raw = trial.signal
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)

    for col in ACCEL_COLUMNS:
        axes[0].plot(raw["time_s"], raw[col], label=f"raw {col}", alpha=0.6)
    axes[0].set_title(
        f"RAW -- {trial.metadata.subject_id} {trial.metadata.activity_code} "
        f"{trial.metadata.trial_id} ({trial.metadata.label})"
    )
    axes[0].set_ylabel("accel (g)")
    axes[0].legend(loc="upper right", fontsize=8)

    for col in ACCEL_COLUMNS:
        axes[1].plot(
            np.arange(len(harmonized)) / 100.0, harmonized[col],
            label=f"harmonized {col}", alpha=0.8,
        )
    axes[1].set_title("HARMONIZED (aligned + 0.5-20 Hz band-pass, 100 Hz)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("accel (g)")
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _impact_frame_check(trial, harmonized: pd.DataFrame) -> Optional[dict]:
    """Peak-magnitude frame near the labeled impact vs. the label itself.

    Reports BOTH the combined 3-axis magnitude peak AND each individual
    axis's own peak, because these can legitimately disagree by tens of
    frames on the same trial. E.g. a forward fall's dominant
    deceleration is horizontal (acc_x), which can peak several hundred
    ms BEFORE the vertical ground-contact transient (acc_z) that a human
    labeler may have keyed the "impact" frame to -- both are real
    physical events in the same fall, not a bug in one or the other.
    Don't treat a large offset on the combined-magnitude peak alone as
    a red flag without checking which axis is driving it.
    """
    meta = trial.metadata
    if meta.label != "fall" or meta.fall_impact_frame is None:
        return None

    accel = harmonized[ACCEL_COLUMNS].to_numpy()
    magnitude = np.linalg.norm(accel, axis=1)

    # Search a window around the labeled impact frame in ORIGINAL sample
    # indexing. Since resampling here is 100->100 Hz (KFall's native
    # rate) this is a no-op resample, so original frame indices line up
    # 1:1 with harmonized-signal indices -- true for KFall specifically,
    # not a general assumption for other datasets later.
    center = meta.fall_impact_frame
    window = 50  # +/- 0.5 s at 100 Hz
    lo = max(0, center - window)
    hi = min(len(magnitude), center + window)
    if lo >= hi:
        return None

    result = {
        "subject_id": meta.subject_id,
        "activity_code": meta.activity_code,
        "trial_id": meta.trial_id,
        "labeled_onset_frame": meta.fall_onset_frame,
        "labeled_impact_frame": meta.fall_impact_frame,
    }

    magnitude_peak_idx = lo + int(np.argmax(magnitude[lo:hi]))
    result["magnitude_peak_frame"] = magnitude_peak_idx
    result["magnitude_peak_offset"] = magnitude_peak_idx - center

    # Per-axis peak (largest absolute value), so a human can tell which
    # axis is driving the combined-magnitude peak and whether a
    # different axis lines up more closely with the labeled impact.
    for i, col in enumerate(ACCEL_COLUMNS):
        axis_window = accel[lo:hi, i]
        axis_peak_local = int(np.argmax(np.abs(axis_window)))
        axis_peak_idx = lo + axis_peak_local
        result[f"{col}_peak_frame"] = axis_peak_idx
        result[f"{col}_peak_offset"] = axis_peak_idx - center
        result[f"{col}_peak_value"] = float(axis_window[axis_peak_local])

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects", nargs="*", default=None,
        help="Restrict to these subject IDs (e.g. SA06 SA07). Default: all found.",
    )
    parser.add_argument(
        "--max-trials-per-subject", type=int, default=3,
        help="Cap on QA-plotted trials per subject (default 3: ~1 ADL, 1 fall, 1 extra).",
    )
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / "configs" / "datasets" / "kfall.yaml")
    sensor_root = REPO_ROOT / cfg.dataset.sensor_root
    label_root = REPO_ROOT / cfg.dataset.label_root

    if not sensor_root.exists():
        print(f"No real data found at {sensor_root}.")
        print(
            "This script requires real KFall sensor data locally "
            "(see PROJECT_CHECKPOINT.md's Kaggle download notes). "
            "Nothing to QA -- exiting."
        )
        return

    trials = load_all_trials(sensor_root, label_root)
    if not trials:
        print(f"{sensor_root} exists but no trials were discovered. Exiting.")
        return

    subjects_filter = set(args.subjects) if args.subjects else None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Two-pass calibration, same as Task 3.10's orchestration ---
    by_subject: dict[str, list] = {}
    for t in trials:
        by_subject.setdefault(t.metadata.subject_id, []).append(t)

    per_subject_calibration = {
        subject_id: calibrate_subject(subject_trials, STANDING_INITIATED_TASK_IDS)
        for subject_id, subject_trials in by_subject.items()
    }
    resolved_calibrations = resolve_group_fallback(per_subject_calibration)

    # --- 2. Calibration source histogram across all real subjects found ---
    source_counts = summarize_calibration_sources(resolved_calibrations)
    print("=" * 60)
    print(f"Subjects found locally: {sorted(by_subject.keys())}")
    print("Calibration source counts (should be mostly T01):")
    for source, count in sorted(source_counts.items()):
        print(f"  {source:>15}: {count}")
    print("=" * 60)

    calib_summary_path = OUTPUT_DIR / "calibration_source_summary.csv"
    pd.DataFrame(
        [{"source": s, "count": c} for s, c in source_counts.items()]
    ).to_csv(calib_summary_path, index=False)
    print(f"Calibration summary written to {calib_summary_path}")

    # --- 1. Raw vs harmonized overlay plots + 3. impact-frame check ---
    selected_trials, _ = _pick_qa_trials(
        trials, subjects_filter, args.max_trials_per_subject
    )

    impact_checks = []
    n_quarantined_in_qa_set = 0

    for trial in selected_trials:
        calibration = resolved_calibrations[trial.metadata.subject_id]
        harmonization_config = HarmonizationConfig()
        harmonized = harmonize_trial(trial, calibration, harmonization_config)

        issues = validate_harmonized_trial(
            harmonized,
            trial.metadata,
            calibration,
            expected_rate_hz=harmonization_config.target_rate_hz,
        )
        if issues:
            n_quarantined_in_qa_set += 1
            print(
                f"  [quarantined, plotting anyway for inspection] "
                f"{trial.metadata.subject_id} {trial.metadata.activity_code} "
                f"{trial.metadata.trial_id}: {issues}"
            )

        out_name = (
            f"{trial.metadata.subject_id}_{trial.metadata.activity_code}_"
            f"{trial.metadata.trial_id}.png"
        )
        _plot_raw_vs_harmonized(trial, harmonized, OUTPUT_DIR / out_name)

        check = _impact_frame_check(trial, harmonized)
        if check is not None:
            check["calibration_source"] = calibration.source
            impact_checks.append(check)

    print(f"Saved {len(selected_trials)} raw-vs-harmonized plots to {OUTPUT_DIR}")
    if n_quarantined_in_qa_set:
        print(
            f"NOTE: {n_quarantined_in_qa_set} of the plotted trials failed "
            "validation and would be quarantined by Task 3.10's pipeline -- "
            "still plotted here so you can see why."
        )

    # --- Impact-frame check summary ---
    if impact_checks:
        impact_df = pd.DataFrame(impact_checks)
        impact_path = OUTPUT_DIR / "impact_frame_check.csv"
        impact_df.to_csv(impact_path, index=False)
        print(
            "\nFall trials: labeled impact frame vs. detected peak frames "
            "(combined-magnitude and per-axis)"
        )
        print(impact_df.to_string(index=False))
        print(f"\nImpact-frame check written to {impact_path}")
        print(
            "Read this by eye against the saved plot, not just the offset "
            "number: the combined-magnitude peak and a given axis's own "
            "peak can legitimately land tens of frames apart on the SAME "
            "trial -- e.g. a forward fall's horizontal deceleration "
            "(acc_x) can peak well before the vertical ground-contact "
            "transient (acc_z) that the label may be keyed to. A large "
            "offset only needs a closer look if the plot does NOT show a "
            "clear, fall-like transient anywhere near the labeled window."
        )
    else:
        print(
            "\nNo fall trials with impact labels were in the selected QA set "
            "-- widen --subjects or --max-trials-per-subject to include one."
        )


if __name__ == "__main__":
    main()
