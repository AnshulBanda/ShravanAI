"""Lightweight config loader with Hydra-style `defaults:` composition,
built on OmegaConf.

This is intentionally NOT full Hydra -- no CLI override sweeps, no
multirun. It just lets one YAML file reference others via a `defaults:`
list (paths relative to the referencing file, no .yaml suffix), so that
e.g. configs/datasets/kfall.yaml can pull in configs/base.yaml without
duplicating paths.project.settings everywhere.

Usage:
    from shared.config import load_config
    cfg = load_config("configs/datasets/kfall.yaml")
    cfg.dataset.sensor_root  # fully resolved, interpolations included
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML config file, resolving any `defaults:` chain.

    Later keys (from the file itself) override earlier ones (from its
    defaults), matching the usual Hydra composition order.
    """
    path = Path(path).resolve()
    return _load(path)


def _load(path: Path) -> DictConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg = OmegaConf.load(path)
    merged = OmegaConf.create({})

    defaults = cfg.pop("defaults", None)
    if defaults:
        for rel in defaults:
            default_path = (path.parent / f"{rel}.yaml").resolve()
            merged = OmegaConf.merge(merged, _load(default_path))

    merged = OmegaConf.merge(merged, cfg)
    return merged
