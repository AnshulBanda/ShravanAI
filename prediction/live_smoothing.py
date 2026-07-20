"""
Turns raw, jittery per-window model output into a STABLE displayed alert
state for a live demo.

Why this exists: `analyze_temporal_errors.py`'s aggregate results showed
the raw per-window prediction genuinely flip-flopping between non_fall /
pre_impact / fall window-to-window, sometimes several times a second
(see e.g. the SA18/T22 trace where predicted label alternates every
1-2 windows in the 500-650 frame range). That's a real property of the
current checkpoint, not something worth hiding -- but showing that raw
flicker on a live demo display would look broken regardless of the
underlying accuracy. This module does NOT change what the model
predicts; it only changes how raw per-window predictions get turned
into a stable, demo-safe displayed state, via two standard real-time
signal-processing techniques:

  1. EMA (exponential moving average) smoothing of the raw class
     probabilities, so a single noisy window can't singlehandedly flip
     the displayed state.
  2. Hysteresis (different "enter" vs "exit" thresholds per class,
     i.e. a Schmitt trigger) so the state can't rapidly oscillate right
     at a threshold boundary -- it takes a clearly higher bar to ESCALATE
     the alert than to STAY at an already-escalated level.

Additionally, once FALL is detected, the alert LATCHES for a fixed
number of windows (`fall_latch_frames`) before it's allowed to clear --
a fall alert disappearing after one calm-looking window would be a
believability problem in a live demo (and arguably a real safety
problem too: a caregiver-facing system shouldn't un-alert a moment
after flagging a fall just because one window's probabilities dipped).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class AlertState(str, Enum):
    CALM = "calm"
    PRE_IMPACT = "pre_impact"
    FALL = "fall"


@dataclass
class SmootherConfig:
    ema_alpha: float = 0.3          # higher = more weight on the newest window, less smoothing
    pre_impact_enter: float = 0.55  # smoothed P(pre_impact) must clear THIS to escalate CALM -> PRE_IMPACT
    pre_impact_exit: float = 0.35   # smoothed P(pre_impact) must drop below THIS to de-escalate back to CALM
    fall_enter: float = 0.50        # smoothed P(fall) must clear THIS to escalate to FALL
    fall_exit: float = 0.30         # smoothed P(fall) must drop below THIS (AFTER the latch expires) to de-escalate
    fall_latch_frames: int = 20     # ~2s at the pipeline's 0.1s stride -- minimum time FALL stays displayed once triggered


class PredictionSmoother:
    """Feed raw per-window class probabilities in, one window at a time
    (in chronological order -- this is a stateful streaming object, not
    a batch function), get a stable AlertState out.

    Usage:
        smoother = PredictionSmoother()
        for raw_probs in stream_of_window_probs:   # raw_probs = [P(non_fall), P(pre_impact), P(fall)]
            state = smoother.update(raw_probs)
            display(state)   # state only changes when the hysteresis/latch rules actually allow it
    """

    def __init__(self, config: SmootherConfig | None = None):
        self.config = config or SmootherConfig()
        self.smoothed_probs: np.ndarray | None = None
        self.state: AlertState = AlertState.CALM
        self._fall_latch_remaining: int = 0

    def reset(self) -> None:
        """Call between trials/sessions -- EMA state and latch must not leak across separate playbacks."""
        self.smoothed_probs = None
        self.state = AlertState.CALM
        self._fall_latch_remaining = 0

    def update(self, raw_probs) -> AlertState:
        raw_probs = np.asarray(raw_probs, dtype=float)
        if raw_probs.shape != (3,):
            raise ValueError(f"Expected raw_probs shape (3,) as [P(non_fall), P(pre_impact), P(fall)], got {raw_probs.shape}")

        cfg = self.config
        if self.smoothed_probs is None:
            self.smoothed_probs = raw_probs.copy()
        else:
            self.smoothed_probs = cfg.ema_alpha * raw_probs + (1 - cfg.ema_alpha) * self.smoothed_probs

        non_fall_p, pre_impact_p, fall_p = self.smoothed_probs

        # Fall alert is latched: once triggered, ignore everything else
        # (including a fresh, higher-confidence fall reading -- doesn't
        # matter, we're already at the highest alert level) until the
        # latch counts down to zero.
        if self._fall_latch_remaining > 0:
            self._fall_latch_remaining -= 1
            self.state = AlertState.FALL
            return self.state

        if fall_p >= cfg.fall_enter:
            self.state = AlertState.FALL
            self._fall_latch_remaining = cfg.fall_latch_frames
        elif self.state == AlertState.FALL:
            # Latch just expired this call (falls through from the block
            # above on a PRIOR call, never this one) -- decide whether to
            # step down to PRE_IMPACT or all the way to CALM using the
            # EXIT thresholds, not the enter thresholds.
            if fall_p >= cfg.fall_exit:
                pass  # stay FALL a bit longer, still elevated
            elif pre_impact_p >= cfg.pre_impact_exit:
                self.state = AlertState.PRE_IMPACT
            else:
                self.state = AlertState.CALM
        elif self.state == AlertState.PRE_IMPACT:
            if pre_impact_p < cfg.pre_impact_exit:
                self.state = AlertState.CALM
            # else: stay PRE_IMPACT, still elevated (hysteresis band)
        else:  # self.state == AlertState.CALM
            if pre_impact_p >= cfg.pre_impact_enter:
                self.state = AlertState.PRE_IMPACT
            # else: stay CALM

        return self.state
