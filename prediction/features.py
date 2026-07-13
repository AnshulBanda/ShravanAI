"""Auxiliary per-sample feature channels for the prediction pipeline.

Per the blueprint's Pipeline 2 §5: given the short window (1.0s) and
dense stride (0.1s), computing aggregate handcrafted features per
window the way `detection/features.py` does (one scalar vector per
window, feeding an XGBoost branch) "would be extremely noisy and
expensive" here. Instead the blueprint calls for a small set of
real-time-cheap features as ROLLING, per-sample AUXILIARY CHANNELS fed
alongside the raw signal into the deep model (ConvLSTM/tiny-Transformer,
not yet built) -- not a separate tabular feature matrix. That's the
key structural difference from detection/features.py, not just a
smaller feature list.

Two channels, per the blueprint:
  - Rolling acceleration magnitude and its first derivative (jerk) --
    "the single most discriminative signal for onset detection."
  - Rolling tilt-angle deviation from the subject's calibrated
    standing baseline.

On the tilt-angle "calibrated standing baseline" wording specifically:
this is NOT an approximation here, unlike it might sound. The
harmonization pipeline's axis-alignment step (`shared/harmonize/
axis_alignment.py`) already rotates every trial's signal so that the
subject's own calibrated standing orientation IS the canonical
vertical (z) axis -- that rotation is derived from a per-subject
stationary-standing calibration segment (see PROJECT_CHECKPOINT.md's
three-tier calibration section). So the angle between the
instantaneous acceleration vector and the canonical z-axis, computed
directly on the already-harmonized signal, already IS "deviation from
the calibrated standing baseline" by construction -- no separate
baseline value needs to be carried through or subtracted. This mirrors
`detection/features.py`'s `tilt_mean`/`tilt_std` computation exactly
(same formula), just kept per-sample here instead of aggregated,
because this pipeline needs a rolling channel, not a window-level
scalar.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from prediction.dataset import CHANNELS, load_window

# Column order for the auxiliary channels this module adds. Appended
# after the 6 raw CHANNELS when building an augmented window (see
# `augment_window`), never interleaved -- keeps "raw" vs. "derived"
# channels unambiguous for whatever model consumes this later.
AUX_CHANNEL_NAMES = ["accel_mag", "jerk", "tilt_deviation_deg"]


def compute_auxiliary_channels(window: np.ndarray, sample_rate_hz: float = 100.0) -> np.ndarray:
    """Compute the rolling auxiliary channels for one window.

    `window` must be shape (n_samples, 6) in CHANNELS order
    (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z) -- exactly what
    `prediction.dataset.load_window` returns.

    Returns shape (n_samples, 3) in AUX_CHANNEL_NAMES order. Every
    output channel is the SAME LENGTH as the input window (one value
    per input sample) -- deliberately, since these are meant to be
    concatenated alongside the raw channels as extra input channels for
    a sequence model, not aggregated into a single scalar per window
    the way `detection/features.py` does. This is why jerk here uses
    `np.gradient` (central differences, same-length output) rather than
    `detection/features.py`'s `np.diff` (which shortens the array by
    one) -- that shortening is fine for an aggregate stat but would
    silently misalign this channel against the raw signal by one
    sample if used here.
    """
    if window.shape[1] != len(CHANNELS):
        raise ValueError(
            f"Expected {len(CHANNELS)} channels ({CHANNELS}), got shape {window.shape}"
        )
    if window.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 samples to compute a rolling derivative (jerk); "
            f"got {window.shape[0]}. Should not occur given the prediction "
            f"pipeline's 100-sample window length, but guarded rather than "
            f"silently producing a degenerate result."
        )

    acc = window[:, :3]

    accel_mag = np.linalg.norm(acc, axis=1)

    # Central-difference derivative, same length as input (np.gradient,
    # not np.diff) -- see docstring above for why same-length matters
    # here specifically.
    jerk = np.gradient(accel_mag) * sample_rate_hz

    # Same formula as detection/features.py's per-window tilt_mean/std,
    # kept per-sample here -- angle (degrees) between the acceleration
    # vector and the canonical vertical (z) axis, which calibration
    # already defined as the subject's standing baseline (see module
    # docstring).
    norm_safe = np.where(accel_mag == 0, 1e-8, accel_mag)
    cos_theta = np.clip(acc[:, 2] / norm_safe, -1.0, 1.0)
    tilt_deviation_deg = np.degrees(np.arccos(cos_theta))

    return np.stack([accel_mag, jerk, tilt_deviation_deg], axis=1).astype(np.float32)


def augment_window(window: np.ndarray, sample_rate_hz: float = 100.0) -> np.ndarray:
    """Concatenate `window`'s 6 raw channels with its 3 auxiliary
    channels, returning shape (n_samples, 9) in `CHANNELS +
    AUX_CHANNEL_NAMES` order.
    """
    aux = compute_auxiliary_channels(window, sample_rate_hz=sample_rate_hz)
    return np.concatenate([window.astype(np.float32), aux], axis=1)


def load_augmented_window(
    window_row: pd.Series,
    window_length_samples: int,
    sample_rate_hz: float = 100.0,
    signal_cache: Optional[dict] = None,
) -> np.ndarray:
    """Load one window's raw signal (via `prediction.dataset.load_window`,
    same edge-padding contract) and augment it with the auxiliary
    channels, in one call -- the function a model's data loader will
    actually call per window. Returns shape (window_length_samples, 9).
    """
    window = load_window(window_row, window_length_samples, signal_cache=signal_cache)
    return augment_window(window, sample_rate_hz=sample_rate_hz)
