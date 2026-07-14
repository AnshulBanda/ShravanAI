"""ConvLSTM branch for the prediction pipeline.

Per blueprint Pipeline 2 §6: "conv blocks for local motion-signature
extraction -> LSTM for temporal integration -> 3-class softmax."
Called out as "the architecture with the strongest directly-comparable
published benchmark on this exact three-class pre-impact task
(~93-96% per-class sensitivity), and it also has the lowest reported
on-device latency" -- relevant for the eventual real-time caregiver-
alert use case.

Conv block channel/kernel sizes (32->64->128, kernels 5/5/3) reused
from `detection/model.py`'s Branch B spec for consistency between the
two pipelines, per the blueprint's stated preference for reusing one
architectural philosophy (DEAF-Net) across both rather than
introducing unrelated conventions pipeline-to-pipeline.

One deliberate DIFFERENCE from detection's branch, worth flagging:
detection's spec calls it "CNN + BiLSTM" (bidirectional). This module
uses a UNIDIRECTIONAL LSTM instead, matching Pipeline 2 §6's literal
wording ("LSTM for temporal integration", no "Bi-"). This isn't purely
a copy-paste omission -- for prediction specifically, a real-time
deployment only ever has access to frames up to the current moment, so
a unidirectional LSTM matches that deployment constraint more
naturally than detection's offline, whole-trial-already-available
BiLSTM does (even though, strictly, a fixed window's bidirectionality
wouldn't itself violate real-time causality once the window is
closed -- the point is this is a genuine architectural choice worth
noting, not an accidental inconsistency between the two branches).
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_INPUT_CHANNELS = 9  # 6 raw (acc/gyro) + 3 auxiliary (accel_mag, jerk, tilt_deviation_deg)
N_CLASSES = 3          # non_fall, pre_impact, fall -- see prediction/labelers.py


class ConvLSTM(nn.Module):
    def __init__(
        self,
        n_input_channels: int = N_INPUT_CHANNELS,
        n_classes: int = N_CLASSES,
        lstm_hidden_size: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        def conv_block(in_ch, out_ch, kernel_size):
            padding = kernel_size // 2  # 'same'-style padding -- length only changes via the pool below, not the conv itself
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )

        self.conv_stack = nn.Sequential(
            conv_block(n_input_channels, 32, kernel_size=5),
            conv_block(32, 64, kernel_size=5),
            conv_block(64, 128, kernel_size=3),
        )
        # 100 samples -> 50 -> 25 -> 12 after 3x MaxPool1d(2) (floor
        # division each time) -- LSTM operates on this length-12,
        # 128-channel sequence, not the original 100 raw samples.
        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_input_channels, seq_len) -- e.g. (B, 9, 100),
        matching `prediction.torch_dataset.PredictionWindowDataset`'s
        output layout directly (no transpose needed by the caller).

        Returns raw logits, shape (batch, n_classes) -- NOT softmaxed;
        pairs with `prediction.losses.FocalLoss` or
        `nn.CrossEntropyLoss`, both of which expect logits, not
        probabilities.
        """
        features = self.conv_stack(x)               # (batch, 128, 12)
        features = features.transpose(1, 2)          # (batch, 12, 128) -- LSTM wants (batch, seq, feature)
        _, (h_n, _) = self.lstm(features)             # h_n: (1, batch, hidden)
        final_hidden = h_n[-1]                        # (batch, hidden) -- last (only, since unidirectional/1-layer) LSTM layer's final hidden state
        return self.classifier(self.dropout(final_hidden))
