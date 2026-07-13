"""Window-boundary generation for the detection pipeline.

Thin re-export of `shared/windowing.py` with detection's own default
config values (2.0s window / 1.0s stride, per the blueprint's Pipeline
1 §3 spec). The boundary-generation logic itself moved to `shared/` so
`prediction/windowing.py` can reuse it with its own (dense) defaults
without importing across the detection/prediction boundary -- see
`shared/windowing.py`'s module docstring for the full rationale.

No behavior change from the pre-move version: same class names, same
field names, same defaults, same function signature. Existing callers
(`detection/dataset.py`, `tests/test_windowing.py`,
`tests/test_detection_dataset.py`) work unchanged.
"""
from __future__ import annotations

from shared.windowing import WindowingConfig, WindowSpec, generate_window_specs

__all__ = ["WindowingConfig", "WindowSpec", "generate_window_specs"]
