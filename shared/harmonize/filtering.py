"""Band-pass filtering.

Applies the frozen Stage 3 filter design: 4th-order Butterworth
band-pass, 0.5-20 Hz, zero-phase (via filtfilt so no time-shift is
introduced -- important since onset/impact frame numbers need to stay
aligned to the original sample indices).

Why 0.5-20 Hz and not the originally-proposed 5 Hz low-pass: a 5 Hz
cutoff was validated in SisFall's own paper for denoising THEIR raw
high-frequency ADC data, but a literature check (sacral-IMU placement,
similar to KFall's lower-back mounting) found that a 5 Hz cutoff only
weakly correlates with true peak impact force, while 10 Hz correlates
moderately -- meaning 5 Hz risks attenuating exactly the fall-impact
transient that's often the most discriminative signal for detection.
0.5-20 Hz keeps the low end for removing slow postural drift and the
high end wide enough to preserve the impact spike. See
test_filtering.py's impact-preservation test for the actual empirical
justification, not just the literature citation.
"""
from __future__ import annotations

import pandas as pd
from scipy.signal import butter, filtfilt


def apply_bandpass_filter(
    signal: pd.DataFrame,
    columns: list[str],
    sample_rate_hz: float,
    low_hz: float = 0.5,
    high_hz: float = 20.0,
    order: int = 4,
) -> pd.DataFrame:
    """Apply a zero-phase Butterworth band-pass filter to the given
    columns of `signal`. Columns not listed are passed through
    unchanged. Returns a new DataFrame; does not mutate the input.
    """
    nyquist = sample_rate_hz / 2.0
    if high_hz >= nyquist:
        raise ValueError(
            f"high_hz={high_hz} must be below the Nyquist frequency "
            f"({nyquist} Hz for sample_rate_hz={sample_rate_hz}); "
            f"choose a lower cutoff or confirm the sample rate is correct."
        )

    b, a = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band")

    out = signal.copy()
    for col in columns:
        out[col] = filtfilt(b, a, signal[col].to_numpy())

    return out
