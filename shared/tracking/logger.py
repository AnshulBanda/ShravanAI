"""Experiment tracking wrapper.

Tries Weights & Biases first; falls back to writing a local JSON/YAML
run record if wandb isn't installed, has no API key configured, or can't
reach the network (this happens on some Kaggle sessions mid-run, and
always in fully offline dev environments). Every run, regardless of
backend, writes results/<project>/<run_name>/ locally -- see blueprint
sec 9, "local fallback": you never want a long training run's results to
depend entirely on an external service's uptime at the moment it finishes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

try:
    import wandb
    _WANDB_IMPORTABLE = True
except ImportError:
    _WANDB_IMPORTABLE = False


class RunLogger:
    def __init__(
        self,
        project: str,
        run_name: str,
        config: DictConfig,
        results_root: str = "results",
        use_wandb: bool = True,
    ):
        self.project = project
        self.run_name = run_name
        self.config = config
        self.run_dir = Path(results_root) / project / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._metrics: list[dict[str, Any]] = []
        self._wandb_run = None

        # Always persist the fully-resolved config locally, regardless of
        # wandb availability -- this is what lets you look back at a
        # results row months later and know exactly what produced it.
        (self.run_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(config))

        if use_wandb and _WANDB_IMPORTABLE:
            try:
                self._wandb_run = wandb.init(
                    project=project,
                    name=run_name,
                    config=OmegaConf.to_container(config, resolve=True),
                )
            except Exception as exc:
                print(f"[RunLogger] wandb unavailable ({exc}); logging locally only.")
                self._wandb_run = None
        elif use_wandb and not _WANDB_IMPORTABLE:
            print("[RunLogger] wandb not installed; logging locally only.")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        record = {"step": step, "timestamp": time.time(), **metrics}
        self._metrics.append(record)
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

    def finish(self) -> None:
        (self.run_dir / "metrics.json").write_text(json.dumps(self._metrics, indent=2))
        if self._wandb_run is not None:
            self._wandb_run.finish()
        print(f"[RunLogger] run '{self.run_name}' recorded at {self.run_dir}")
