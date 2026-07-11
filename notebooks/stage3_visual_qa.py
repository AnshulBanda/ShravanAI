"""Visual QA pass (human-in-the-loop, NOT automated).

Originally written for Stage 3/Task 3.11 (KFall only); generalized
during Stage 5 to work against any registered dataset via --dataset,
reusing orchestration.py's shared `get_trial_loader`/`resolve_calibrations`
helpers instead of the KFall-only calibration logic this script used to
duplicate inline. That duplication is exactly how a bug (calibration
running on the raw, unconverted signal) stayed fixed in
orchestration.py but not here, until this refactor -- see
PROJECT_CHECKPOINT.md's Stage 5 section for the full story.

This is a one-off exploratory script, not imported by anything and not
part of the test suite. Its entire purpose is to let a human eyeball
harmonization quality on REAL data before trusting a dataset's
harmonization output. It intentionally does not assert/fail on
anything -- read the printed output and the saved plots yourself.

Usage:
    python notebooks/stage3_visual_qa.py --dataset kfall
    python notebooks/stage3_visual_qa.py --dataset sisfall
    python notebooks/stage3_visual_qa.py --dataset kfall --subjects SA06 SA07
    python notebooks/stage3_visual_qa.py --dataset sisfall --max-trials-per-subject 3

Requires real data locally under the path each dataset's
configs/datasets/<name>.yaml points to. Degrades gracefully to
whatever subjects are actually present.

What this produces:
1. Raw-vs-harmonized signal overlay plots for a handful of real trials,
   spread across whichever subjects/activity types are available.
2. A histogram/table of calibration source across all real subjects
   found for the chosen dataset (KFall: should be mostly "T01". SisFall:
   should be "auto_detected"/"group_fallback" only -- "T01" appearing
   here for SisFall would indicate a real regression, since SisFall has
   no dedicated calibration trial by design).
3. For every real fall trial with labeled onset/impact frames
   (currently KFall only -- SisFall has none, see readers_sisfall.py),
   the labeled frames vs. the frame of peak harmonized-signal magnitude
   near impact.

All outputs are written under results/stage3_visual_qa/<dataset>/ so
different datasets' QA runs don't overwrite each other.
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
from shared.harmonize.axis_alignment import summarize_calibration_sources
from shared.harmonize.orchestration import get_trial_loader, resolve_calibrations
from shared.harmonize.pipeline import HarmonizationConfig, harmonize_trial
from shared.harmonize.units import ACCEL_COLUMNS
from shared.harmonize.validation import validate_harmonized_trial

REPO_ROOT = Path(__file__).parent.parent


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


def _plot_raw_vs_harmonized(
    trial, harmonized: pd.DataFrame, target_rate_hz: float, out_path: Path
) -> None:
    raw = trial.signal
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)

    # Raw signal's own accel columns vary by dataset (KFall: acc_x/y/z
    # already; SisFall: raw_adxl_acc_x/y/z, not yet unit-converted) --
    # plot whatever accel-like columns are actually present rather than
    # assuming ACCEL_COLUMNS exists pre-conversion.
    raw_accel_cols = [c for c in raw.columns if "acc" in c.lower() and "mma" not in c.lower()]
    for col in raw_accel_cols:
        axes[0].plot(raw["time_s"], raw[col], label=f"raw {col}", alpha=0.6)
    axes[0].set_title(
        f"RAW -- {trial.metadata.subject_id} {trial.metadata.activity_code} "
        f"{trial.metadata.trial_id} ({trial.metadata.label})"
    )
    axes[0].set_ylabel("accel (raw units)")
    axes[0].legend(loc="upper right", fontsize=8)

    for col in ACCEL_COLUMNS:
        axes[1].plot(
            np.arange(len(harmonized)) / target_rate_hz, harmonized[col],
            label=f"harmonized {col}", alpha=0.8,
        )
    axes[1].set_title(f"HARMONIZED (aligned + band-pass filtered, {target_rate_hz:.0f} Hz)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("accel (g)")
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _impact_frame_check(trial, harmonized: pd.DataFrame) -> Optional[dict]:
    """Peak-magnitude frame near the labeled impact vs. the label itself.

    In practice this only ever produces output for KFall trials -- it's
    the `meta.fall_impact_frame is None` early return below that makes
    that true, not a `dataset ==` check, since SisFall (and any future
    dataset without frame-level fall labels) always has `fall_impact_frame
    = None` by construction (see readers_sisfall.py's docstring).

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
    # indexing. Assumes the labeled frame index and the harmonized
    # signal's index both refer to the SAME sample rate -- true for
    # KFall (100 Hz native == 100 Hz target, no-op resample). If a
    # future dataset ever has frame-level labels AND native_rate_hz !=
    # target_rate_hz, `center` below would need rescaling by that
    # ratio first -- not needed yet since no such dataset exists.
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
        "--dataset", required=True, choices=["kfall", "sisfall"],
        help="Which dataset to QA.",
    )
    parser.add_argument(
        "--subjects", nargs="*", default=None,
        help="Restrict to these subject IDs (e.g. SA06 SA07). Default: all found.",
    )
    parser.add_argument(
        "--max-trials-per-subject", type=int, default=3,
        help="Cap on QA-plotted trials per subject (default 3: ~1 ADL, 1 fall, 1 extra).",
    )
    args = parser.parse_args()
    dataset = args.dataset
    output_dir = REPO_ROOT / "results" / "stage3_visual_qa" / dataset

    cfg = load_config(REPO_ROOT / "configs" / "datasets" / f"{dataset}.yaml")
    sensor_root = REPO_ROOT / cfg.dataset.sensor_root
    label_root = REPO_ROOT / cfg.dataset.label_root if cfg.dataset.label_root else None

    if not sensor_root.exists():
        print(f"No real data found at {sensor_root}.")
        print(
            f"This script requires real {dataset} sensor data locally. "
            "Nothing to QA -- exiting."
        )
        return

    trials = get_trial_loader(dataset)(sensor_root, label_root)
    if not trials:
        print(f"{sensor_root} exists but no trials were discovered. Exiting.")
        return

    subjects_filter = set(args.subjects) if args.subjects else None
    output_dir.mkdir(parents=True, exist_ok=True)

    by_subject: dict[str, list] = {}
    for t in trials:
        by_subject.setdefault(t.metadata.subject_id, []).append(t)

    # --- Two-pass calibration, shared with orchestration.run_harmonization ---
    resolved_calibrations = resolve_calibrations(dataset, trials)

    # --- 2. Calibration source histogram across all real subjects found ---
    source_counts = summarize_calibration_sources(resolved_calibrations)
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Subjects found locally: {sorted(by_subject.keys())}")
    if dataset == "kfall":
        print("Calibration source counts (should be mostly T01):")
    else:
        print(
            "Calibration source counts (should be ONLY auto_detected / "
            f"group_fallback -- {dataset} has no dedicated calibration "
            "trial, so a 'T01' entry here would indicate a real "
            "regression, not a good sign):"
        )
    for source, count in sorted(source_counts.items()):
        print(f"  {source:>15}: {count}")
    print("=" * 60)

    calib_summary_path = output_dir / "calibration_source_summary.csv"
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
    harmonization_config = HarmonizationConfig()

    for trial in selected_trials:
        calibration = resolved_calibrations[trial.metadata.subject_id]
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
        _plot_raw_vs_harmonized(
            trial, harmonized, harmonization_config.target_rate_hz, output_dir / out_name
        )

        check = _impact_frame_check(trial, harmonized)
        if check is not None:
            check["calibration_source"] = calibration.source
            impact_checks.append(check)

    print(f"Saved {len(selected_trials)} raw-vs-harmonized plots to {output_dir}")
    if n_quarantined_in_qa_set:
        print(
            f"NOTE: {n_quarantined_in_qa_set} of the plotted trials failed "
            "validation and would be quarantined by Task 3.10's pipeline -- "
            "still plotted here so you can see why."
        )

    # --- Impact-frame check summary ---
    if impact_checks:
        impact_df = pd.DataFrame(impact_checks)
        impact_path = output_dir / "impact_frame_check.csv"
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
