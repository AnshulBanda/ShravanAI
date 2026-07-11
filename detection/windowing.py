"""Window-boundary generation for the detection pipeline.

Pure logic, no I/O -- given a trial's sample count and a windowing
config, produces the list of (start, end) frame boundaries plus
padding bookkeeping. Kept separate from dataset.py (which does the
manifest-querying and file-loading side) the same way
readers_*.py/units.py/resample.py were kept as separate single-purpose
modules during harmonization -- easier to unit test in isolation.

Per the blueprint's Pipeline 1 spec (§3): 2.0s windows (200 samples
@ 100Hz), 1.0s stride (50% overlap), never crossing a trial boundary,
padding short trials rather than dropping them.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WindowingConfig:
    window_length_s: float = 2.0
    stride_s: float = 1.0
    target_rate_hz: float = 100.0  # matches harmonization's target rate; not independently configurable, since windowing always operates on already-harmonized (100Hz) signals

    @property
    def window_length_samples(self) -> int:
        return round(self.window_length_s * self.target_rate_hz)

    @property
    def stride_samples(self) -> int:
        return round(self.stride_s * self.target_rate_hz)


@dataclass
class WindowSpec:
    start_frame: int
    end_frame: int          # exclusive; end_frame - start_frame == n_real_samples (may be < window_length_samples for a short/trailing window, BEFORE padding)
    n_real_samples: int
    n_pad_samples: int      # window_length_samples - n_real_samples; 0 for a full, unpadded window


def generate_window_specs(trial_n_samples: int, config: WindowingConfig) -> list[WindowSpec]:
    """Generate window boundaries covering a trial of `trial_n_samples`.

    Design decisions, made explicit here rather than left implicit:
    - The ENTIRE trial is always covered, start to end. A short trial
      (shorter than one window) still produces exactly one window,
      padded. A longer trial's trailing leftover after the last
      full-stride window (if any) still produces one final padded
      window, rather than being silently dropped -- this matters for
      fall detection specifically, since a fall event can occur near
      the end of a short trial file, and dropping that tail would mean
      dropping the actual fall.
    - Padding amount is reported (`n_pad_samples`) but the padding
      VALUES themselves are not decided here -- see dataset.py's
      `load_window` for the edge-padding strategy. This function only
      decides boundaries.
    - Returns an empty list for `trial_n_samples <= 0` (defensive; a
      trial with 0 real samples has nothing to window).
    """
    if trial_n_samples <= 0:
        return []

    window_length = config.window_length_samples
    stride = config.stride_samples

    if trial_n_samples <= window_length:
        return [WindowSpec(
            start_frame=0,
            end_frame=trial_n_samples,
            n_real_samples=trial_n_samples,
            n_pad_samples=window_length - trial_n_samples,
        )]

    specs: list[WindowSpec] = []
    start = 0
    last_covered_end = 0
    while start + window_length <= trial_n_samples:
        end = start + window_length
        specs.append(WindowSpec(
            start_frame=start,
            end_frame=end,
            n_real_samples=window_length,
            n_pad_samples=0,
        ))
        last_covered_end = end
        start += stride

    # Trailing leftover after the last full-stride window -- cover it
    # with one final padded window rather than dropping it, per the
    # design decision above. Starts exactly where the last full window
    # ENDED (`last_covered_end`), not at the next stride position --
    # using the stride position here would either skip real samples
    # (if the next stride lands past last_covered_end, leaving a gap)
    # or, at an exact boundary (last_covered_end == trial_n_samples),
    # incorrectly add a redundant overlapping window when nothing is
    # actually left to cover.
    if last_covered_end < trial_n_samples:
        specs.append(WindowSpec(
            start_frame=last_covered_end,
            end_frame=trial_n_samples,
            n_real_samples=trial_n_samples - last_covered_end,
            n_pad_samples=window_length - (trial_n_samples - last_covered_end),
        ))

    return specs
