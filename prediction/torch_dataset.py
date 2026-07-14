"""PyTorch data-loading layer for the prediction pipeline's model
branches (ConvLSTM / tiny-Transformer -- not yet built).

Two pieces, kept separate on purpose:
  - `PredictionWindowDataset`: turns one row of the windows manifest
    into one (x, y) tensor pair. Ordinary map-style Dataset.
  - `TrialGroupedBatchSampler`: controls WHICH indices end up in the
    same minibatch. This is the "sequence-aware batching" from
    blueprint §7 -- shuffles at the TRIAL level (all of a trial's
    densely-overlapping windows travel together through batch
    construction) rather than shuffling individual windows.

Worth being precise about what this batch sampler does and doesn't do,
since blueprint §7's wording ("to avoid near-duplicate windows leaking
across train/val") could be misread as a leakage-prevention mechanism:
it ISN'T one. Leakage prevention is `prediction.loso`'s job (subject-
level fold splitting, done BEFORE any Dataset/DataLoader exists). This
sampler only affects training dynamics *within* an already-clean
training set -- e.g. avoiding a batch-norm layer computing statistics
over a batch that happens to be all-one-trial while another batch has
none of it. Two different problems; solved by two different modules,
not blurred into one.
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from prediction.features import load_augmented_window


class PredictionWindowDataset(Dataset):
    """One item = one window's augmented (9-channel) signal + its label.

    `x` is returned channel-first, shape (9, window_length_samples) --
    the layout `torch.nn.Conv1d` expects (batch, channels, length), not
    (batch, length, channels).

    Maintains an INSTANCE-level cache of harmonized parquet files
    (keyed by path, shared across `__getitem__` calls) since the
    dense-stride windowing means many consecutive windows come from
    the same trial file -- without this cache, that file would be
    re-read from disk on nearly every call. Known tradeoff, not a bug:
    with a multi-worker `DataLoader` (`num_workers > 0`), each worker
    process gets its OWN copy of this cache (no cross-process sharing),
    so memory use scales with worker count. Fine at KFall's scale
    (5,075 trials, each a small parquet file), worth revisiting only if
    this pipeline is ever pointed at much larger harmonized data.
    """

    def __init__(
        self,
        windows_df: pd.DataFrame,
        window_length_samples: int = 100,
        sample_rate_hz: float = 100.0,
    ):
        self.windows_df = windows_df.reset_index(drop=True)
        self.window_length_samples = window_length_samples
        self.sample_rate_hz = sample_rate_hz
        self._signal_cache: dict = {}

    def __len__(self) -> int:
        return len(self.windows_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.windows_df.iloc[idx]
        augmented = load_augmented_window(
            row,
            self.window_length_samples,
            sample_rate_hz=self.sample_rate_hz,
            signal_cache=self._signal_cache,
        )
        # augmented is (n_samples, 9); transpose to channel-first and
        # force a contiguous copy (the transpose alone is a view with
        # non-contiguous strides, which torch.from_numpy would still
        # accept but which silently costs a copy later inside the
        # model anyway -- doing it once here, explicitly, is cheaper
        # and clearer than an implicit copy inside every forward pass).
        x = torch.from_numpy(np.ascontiguousarray(augmented.T))
        y = torch.tensor(int(row["label_id"]), dtype=torch.long)
        return x, y


class TrialGroupedBatchSampler(Sampler):
    """Yields batches of row-positions, shuffled at the TRIAL level
    (grouped by `harmonized_path`, one group per trial file) rather
    than shuffling individual window rows.

    Within a trial group, window order is preserved (temporal order,
    as they appear in `windows_df`) -- only the ORDER OF GROUPS is
    shuffled between epochs. A batch may still span a boundary between
    two trial groups (this sampler doesn't force batch_size to align
    with trial lengths, which vary), but consecutive windows from one
    trial land in the same or adjacent batches rather than being
    scattered uniformly across the whole epoch.
    """

    def __init__(
        self,
        windows_df: pd.DataFrame,
        batch_size: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        windows_df = windows_df.reset_index(drop=True)
        # groupby(...).indices maps group-key -> positional row indices,
        # already in the DataFrame's original (temporal) row order
        # within each group.
        self._groups: list[np.ndarray] = list(
            windows_df.groupby("harmonized_path", sort=False).indices.values()
        )
        self._total = sum(len(g) for g in self._groups)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed)
        group_order = np.arange(len(self._groups))
        if self.shuffle:
            rng.shuffle(group_order)

        all_indices = np.concatenate([self._groups[g] for g in group_order])
        for start in range(0, len(all_indices), self.batch_size):
            yield all_indices[start : start + self.batch_size].tolist()

    def __len__(self) -> int:
        return (self._total + self.batch_size - 1) // self.batch_size
