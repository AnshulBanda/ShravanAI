"""Stage 5 reader verification script -- run against REAL SisFall data.

The reader (readers_sisfall.py) was built and unit-tested against
synthetic fixtures that match the documented real format, but has only
been spot-checked against 3 real files by hand so far. SisFall has
~4,505 real files across 38 subjects -- this script actually loads all
of them and reports anything that doesn't match expectations, the same
role scripts/verify_setup.py played for KFall's Stage 2.

Usage:
    python scripts/verify_sisfall_reader.py
"""
from collections import Counter
from pathlib import Path

from shared.io.readers_sisfall import discover_trials, load_trial

REPO_ROOT = Path(__file__).parent.parent
SENSOR_ROOT = REPO_ROOT / "data" / "raw" / "sisfall" / "SisFall_dataset"


def main() -> None:
    if not SENSOR_ROOT.exists():
        print(f"No real data found at {SENSOR_ROOT}. Nothing to verify.")
        return

    trial_paths = discover_trials(SENSOR_ROOT)
    print(f"Discovered {len(trial_paths)} trial files under {SENSOR_ROOT}")

    n_loaded = 0
    n_failed = 0
    failures: list[str] = []
    subjects: set[str] = set()
    labels: Counter = Counter()
    row_counts: list[int] = []

    for path in trial_paths:
        try:
            trial = load_trial(path)
        except ValueError as exc:
            n_failed += 1
            failures.append(f"{path.name}: {exc}")
            continue

        n_loaded += 1
        subjects.add(trial.metadata.subject_id)
        labels[trial.metadata.label] += 1
        row_counts.append(len(trial.signal))

    print(f"\nLoaded successfully: {n_loaded}")
    print(f"Failed to load:      {n_failed}")
    if failures:
        print("\nFirst 10 failures (fix the reader or flag these files as bad):")
        for f in failures[:10]:
            print(f"  {f}")

    print(f"\nSubjects found: {len(subjects)} (expect 38: SA01-23, SE01-15)")
    missing = ({f"SA{i:02d}" for i in range(1, 24)} | {f"SE{i:02d}" for i in range(1, 16)}) - subjects
    if missing:
        print(f"  MISSING subjects: {sorted(missing)}")

    print(f"\nLabel counts: {dict(labels)}")
    print(
        f"Row counts: min={min(row_counts)}, max={max(row_counts)}, "
        f"mean={sum(row_counts) / len(row_counts):.0f} "
        f"(at 200 Hz, expect roughly 12s-100s trials -> ~2400-20000 rows, "
        f"per the Readme's per-activity durations)"
    )

    if n_failed == 0 and not missing:
        print("\nAll real SisFall files loaded cleanly. Reader looks solid.")
    else:
        print(
            "\nSome files failed or subjects are missing -- investigate before "
            "trusting this reader for harmonization."
        )


if __name__ == "__main__":
    main()
