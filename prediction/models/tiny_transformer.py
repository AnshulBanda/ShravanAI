"""Tiny-Transformer branch for the prediction pipeline.

Per blueprint Pipeline 2 §6: "Lightweight Transformer (ViT-tiny
style): treats short IMU windows as a small 'image' (channels x time),
self-attention over the time axis." Sized to match the blueprint's
explicit reference point -- "a tiny ViT teacher (3 layers, 3 heads)"
from the PreFallKD pre-impact work -- rather than an arbitrarily chosen
size. `d_model=48` was chosen (not stated in the blueprint) purely
because it's the smallest multiple of `n_heads=3` that still gives
each attention head a reasonable head_dim (16) -- kept small
deliberately, matching "tiny" in both the blueprint's wording and the
PreFallKD reference's own emphasis on a latency-constrained student/
teacher size, not a full-scale transformer.

Each of the 100 timesteps is treated as one token (a 9-dim vector:
6 raw channels + 3 auxiliary), linearly projected to `d_model`, plus a
prepended learnable CLS token (standard ViT classification pattern) --
its output embedding after the encoder stack is what feeds the
classifier head, not a mean-pool over all 100 timestep tokens.

Positional encoding: a learned per-position embedding (one vector per
of the 101 positions -- 100 timesteps + 1 CLS), not a fixed sinusoidal
one. Chosen because window length is fixed (always exactly 100
samples, by construction of `prediction.windowing`) and small, so
there's no need for a sinusoidal encoding's ability to generalize to
unseen sequence lengths -- a learned table is simpler and has been
shown to perform comparably at this scale.
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_INPUT_CHANNELS = 9
N_CLASSES = 3
SEQ_LEN = 100  # matches prediction.windowing.PredictionWindowingConfig's 1.0s window @ 100Hz -- not independently configurable, since the learned positional embedding table below is sized to it


class TinyTransformer(nn.Module):
    def __init__(
        self,
        n_input_channels: int = N_INPUT_CHANNELS,
        n_classes: int = N_CLASSES,
        seq_len: int = SEQ_LEN,
        d_model: int = 48,
        n_heads: int = 3,
        n_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.token_projection = nn.Linear(n_input_channels, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.positional_embedding = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_input_channels, seq_len) -- e.g. (B, 9, 100),
        same channel-first layout as `PredictionWindowDataset`'s
        output and `ConvLSTM.forward`'s input (kept consistent between
        the two branches so a training loop can swap one for the other
        without touching the data pipeline).

        Returns raw logits, shape (batch, n_classes) -- see
        `ConvLSTM.forward`'s docstring for the same logits-not-
        probabilities note; applies identically here.
        """
        batch_size = x.shape[0]
        tokens = x.transpose(1, 2)                      # (batch, seq_len, n_input_channels) -- one token per timestep
        tokens = self.token_projection(tokens)            # (batch, seq_len, d_model)

        cls = self.cls_token.expand(batch_size, -1, -1)   # (batch, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)           # (batch, seq_len+1, d_model)
        tokens = tokens + self.positional_embedding

        encoded = self.encoder(tokens)                     # (batch, seq_len+1, d_model)
        cls_output = encoded[:, 0, :]                        # (batch, d_model) -- the CLS token's own output embedding, not a pooled average over all timesteps
        return self.classifier(self.dropout(cls_output))
