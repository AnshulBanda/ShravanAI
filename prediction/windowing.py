"""Window-boundary generation for the prediction pipeline.

Re-exports `shared/windowing.py`'s generic boundary logic (identical
math to detection's) but with prediction's own defaults: **1.0s window
(100 samples @ 100Hz), 0.1s stride (10-sample dense overlap)**, per the
blueprint's Pipeline 2 §3 spec -- not detection's 2.0s/1.0s.

Why so much denser than detection: the metric this pipeline is judged
on is LEAD TIME (how many ms before impact the model first raises
`pre_impact`), which needs fine-grained temporal resolution to measure
meaningfully. A 1.0s-stride window (detection's stride) would only let
lead time be measured in ~1s increments -- too coarse when KFall falls
themselves are only ~0.6-1.0s from onset to impact (per the blueprint).
1.0s window length is also close to the ceiling of usable length for
the same reason: longer risks a single window spanning both pre-fall
walking and the fall itself, muddying the label (blueprint §3).

`PredictionWindowingConfig` is a thin subclass purely to carry
different defaults -- `shared.windowing.generate_window_specs` doesn't
care which one it's handed, since the boundary math has no pipeline-
specific assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass

from shared.windowing import WindowingConfig, WindowSpec, generate_window_specs

__all__ = ["PredictionWindowingConfig", "WindowSpec", "generate_window_specs"]


@dataclass
class PredictionWindowingConfig(WindowingConfig):
    window_length_s: float = 1.0
    stride_s: float = 0.1
    target_rate_hz: float = 100.0
