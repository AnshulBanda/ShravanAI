"""Resampling with mandatory anti-aliasing.

This module resamples every signal channel to a common target rate
using polyphase resampling (`scipy.signal.resample_poly`), which applies
its own designed anti-aliasing low-pass filter before decimation -- this
is what actually prevents high-frequency content from folding back into
the signal as spurious low-frequency artifacts when downsampling.

For KFall (100 -> 100 Hz) this is a verified no-op, same pattern as
Task 3.1's unit conversion. The real payoff comes with SisFall's native
200 Hz (and FallAllD's, once its exact rate is confirmed), both of which
will go through the same function.

Only downsampling is supported -- upsampling isn't a case this pipeline
needs, and silently "supporting" it via interpolation would manufacture
frequency content that was never in the original signal.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.signal import resample_poly


def resample_signal(
    signal: pd.DataFrame,
    native_rate_hz: float,
    target_rate_hz: float,
    time_col: str = "time_s",
) -> pd.DataFrame:
    """Resample every non-time column of `signal` from native_rate_hz to
    target_rate_hz, with anti-aliasing applied automatically.

    Raises ValueError if target_rate_hz > native_rate_hz (upsampling).
    Returns a new DataFrame; does not mutate the input.
    """
    if target_rate_hz > native_rate_hz:
        raise ValueError(
            f"resample_signal only supports downsampling, got "
            f"native_rate_hz={native_rate_hz} < target_rate_hz={target_rate_hz}"
        )

    if native_rate_hz == target_rate_hz:
        # Verified no-op: return an exact copy rather than routing
        # through resample_poly at all, so there's no ambiguity about
        # whether a 1:1 ratio could introduce floating-point noise.
        return signal.copy()

    ratio = Fraction(target_rate_hz / native_rate_hz).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator

    data_cols = [c for c in signal.columns if c != time_col]
    n_native = len(signal)
    n_target = int(round(n_native * up / down))

    resampled_data = {
        col: resample_poly(signal[col].to_numpy(), up, down)
        for col in data_cols
    }

    # resample_poly's output length can be off by one sample from the
    # exact target length due to rounding -- trim/pad to the expected
    # length so every channel and the reconstructed time axis agree.
    for col in data_cols:
        arr = resampled_data[col]
        if len(arr) > n_target:
            resampled_data[col] = arr[:n_target]
        elif len(arr) < n_target:
            resampled_data[col] = np.pad(arr, (0, n_target - len(arr)), mode="edge")

    start_time = float(signal[time_col].iloc[0]) if time_col in signal.columns else 0.0
    new_time = start_time + np.arange(n_target) / target_rate_hz

    out = pd.DataFrame({time_col: new_time, **resampled_data})
    return out[[time_col] + data_cols]
