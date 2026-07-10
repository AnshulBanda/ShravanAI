"""CLI entry point: harmonize a dataset end-to-end.

No logic lives here -- this just resolves paths from config and calls
shared.harmonize.orchestration.run_harmonization, per this repo's
scripts-are-thin convention.

Usage:
    python scripts/harmonize_dataset.py --dataset kfall
"""
import argparse
from pathlib import Path

from shared.config import load_config
from shared.harmonize.orchestration import run_harmonization
from shared.harmonize.pipeline import HarmonizationConfig

REPO_ROOT = Path(__file__).parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Harmonize a dataset end-to-end.")
    parser.add_argument("--dataset", required=True, choices=["kfall", "sisfall"])
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / "configs" / "datasets" / f"{args.dataset}.yaml")

    sensor_root = REPO_ROOT / cfg.dataset.sensor_root
    # label_root is optional -- SisFall has no separate label file at
    # all (see configs/datasets/sisfall.yaml), so its config sets this
    # to null/None. Resolving `REPO_ROOT / None` would crash, so only
    # resolve it when a real path is actually configured.
    label_root = REPO_ROOT / cfg.dataset.label_root if cfg.dataset.label_root else None
    harmonized_root = REPO_ROOT / cfg.paths.harmonized
    quarantine_root = harmonized_root.parent / "_quarantine"
    manifest_path = harmonized_root / "manifest.parquet"

    summary = run_harmonization(
        dataset=args.dataset,
        sensor_root=sensor_root,
        label_root=label_root,
        harmonized_root=harmonized_root,
        quarantine_root=quarantine_root,
        harmonization_config=HarmonizationConfig(),
        manifest_path=manifest_path,
    )

    print(f"Dataset:                {args.dataset}")
    print(f"Total trials processed: {summary.n_trials_total}")
    print(f"Written (accepted):     {summary.n_written}")
    print(f"Quarantined:            {summary.n_quarantined}")
    print(f"Calibration sources:    {summary.calibration_source_counts}")
    print(f"Manifest written to:    {manifest_path}")


if __name__ == "__main__":
    main()
