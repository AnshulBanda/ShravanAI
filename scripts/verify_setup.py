"""Stage 1+2 verification script.

Run this after cloning the repo to confirm the config loader, logger,
and KFall reader all work end-to-end before writing any more code.
This intentionally uses the synthetic test fixtures, not real KFall
data -- point it at data/raw/kfall once you have the real dataset by
editing SENSOR_ROOT/LABEL_ROOT below, or just run pytest against real
data with a new fixture-free test file (see tests/test_kfall_reader.py
docstring for the suggested split).

Usage:
    python scripts/verify_setup.py
"""
from pathlib import Path

from shared.config import load_config
from shared.io.readers_kfall import load_all_trials
from shared.tracking.logger import RunLogger

REPO_ROOT = Path(__file__).parent.parent
FIXTURE_SENSOR_ROOT = REPO_ROOT / "tests" / "fixtures" / "kfall_mock" / "sensor_data"
FIXTURE_LABEL_ROOT = REPO_ROOT / "tests" / "fixtures" / "kfall_mock" / "label_data"


def main() -> None:
    print("== Stage 1: config loader ==")
    cfg = load_config(REPO_ROOT / "configs" / "datasets" / "kfall.yaml")
    print(f"  project.seed = {cfg.project.seed}")
    print(f"  dataset.sensor_root (interpolated) = {cfg.dataset.sensor_root}")

    print("\n== Stage 1: run logger ==")
    logger = RunLogger(
        project="setup-verification",
        run_name="stage-1-2-check",
        config=cfg,
        results_root=str(REPO_ROOT / "results"),
    )
    logger.log({"status": "config_loaded"}, step=0)

    print("\n== Stage 2: KFall reader (synthetic fixtures) ==")
    trials = load_all_trials(FIXTURE_SENSOR_ROOT, FIXTURE_LABEL_ROOT)
    print(f"  loaded {len(trials)} trials")
    for trial in trials:
        m = trial.metadata
        print(
            f"    subject={m.subject_id} task={m.activity_code} trial={m.trial_id} "
            f"label={m.label} onset={m.fall_onset_frame} impact={m.fall_impact_frame} "
            f"n_samples={len(trial.signal)}"
        )

    logger.log({"status": "kfall_reader_verified", "n_trials": len(trials)}, step=1)
    logger.finish()

    print("\nAll stage 1+2 checks passed.")


if __name__ == "__main__":
    main()
