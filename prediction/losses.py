"""Focal loss for the prediction pipeline's model branches.

Per blueprint Pipeline 2 §7: "Loss: focal loss again, more
aggressively weighted than Pipeline 1 -- pre-impact is the rarest
class and the one false negatives on matter most clinically." ("Focal
loss again" implies detection also used it -- for a binary problem via
`scale_pos_weight`-style class weighting; this module is the 3-class
generalization, not a copy of detection's loss code, since
`detection/model.py` is XGBoost-based and doesn't have a torch loss
module at all.)

Standard multi-class focal loss (Lin et al. 2017): down-weights
easy/confident examples via a `(1 - p_t)^gamma` modulating factor, so
the rare `pre_impact` class doesn't get drowned out by the abundant
`non_fall`/`fall` classes the way plain cross-entropy would let happen.

The "more aggressively weighted" per-class alpha weighting is NOT
given exact numbers in the blueprint -- `default_alpha_weights()`
below is a real design decision, made explicit rather than silently
picked: inverse-frequency weighting (rarer class -> higher weight,
the standard starting point) PLUS an extra multiplier specifically on
`pre_impact` on top of what inverse-frequency alone would give it,
per the blueprint's explicit clinical-priority statement above. Default
extra multiplier (2.0x) is a reasonable starting point, not a value
derived from anything in the blueprint or the published literature --
worth tuning against a real validation confusion matrix once training
is running, not treated as final.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from prediction.labelers import LABEL_TO_INT, PRE_IMPACT


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        alpha: optional per-class weight tensor, shape (n_classes,).
            If None, behaves as unweighted focal loss (all classes
            equal weight) -- NOT the recommended default for training
            on this pipeline's real class imbalance; see
            `default_alpha_weights()` below for the intended default.
        gamma: focusing parameter. 0 reduces to (optionally weighted)
            standard cross-entropy; higher values down-weight easy
            examples more aggressively. 2.0 is the standard default
            from the original focal loss paper, not specific to this
            project.
        reduction: 'mean', 'sum', or 'none' (per-sample losses,
            useful for inspecting which windows the model finds
            hardest during debugging).
        """
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none'; got {reduction!r}")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: (batch, n_classes) raw scores (NOT softmaxed) --
        matches `ConvLSTM`/`TinyTransformer`'s `.forward()` output
        directly. targets: (batch,) int class indices.
        """
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        focal_term = (1.0 - pt) ** self.gamma
        loss = -focal_term * log_pt

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            loss = loss * alpha[targets]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def default_alpha_weights(
    label_counts: dict[str, int],
    pre_impact_extra_boost: float = 2.0,
) -> torch.Tensor:
    """Build a per-class alpha weight tensor from real observed class
    counts (e.g. `windows_df['label'].value_counts().to_dict()` on the
    TRAINING split only -- never on the full dataset including the
    held-out LOSO test subject, which would leak class-balance
    information about the test fold into the loss weighting).

    Returns a tensor of length 3, indexed by `LABEL_TO_INT` (so it can
    be passed straight to `FocalLoss(alpha=...)`), normalized to sum
    to `n_classes` (keeps the overall loss scale comparable to
    unweighted focal loss, rather than the total loss magnitude
    silently drifting with whatever raw counts happen to be passed in).
    """
    for label in LABEL_TO_INT:
        if label not in label_counts or label_counts[label] <= 0:
            raise ValueError(
                f"label_counts must have a positive count for every class "
                f"in LABEL_TO_INT; missing or zero for {label!r}. Got: "
                f"{label_counts}. (A LOSO fold with zero windows of some "
                f"class in its TRAINING split would be a real, worth-"
                f"investigating data issue, not something to silently "
                f"paper over with a default weight.)"
            )

    inverse_freq = {label: 1.0 / count for label, count in label_counts.items()}
    inverse_freq[PRE_IMPACT] *= pre_impact_extra_boost

    weights = torch.zeros(len(LABEL_TO_INT))
    for label, class_id in LABEL_TO_INT.items():
        weights[class_id] = inverse_freq[label]

    # Normalize to sum to n_classes (not to 1.0) -- keeps each
    # individual weight centered around ~1.0 on average, so the
    # loss's overall magnitude stays comparable to gamma-only focal
    # loss with alpha=None, rather than shrinking by a factor of
    # n_classes for no principled reason.
    weights = weights * (len(LABEL_TO_INT) / weights.sum())
    return weights
