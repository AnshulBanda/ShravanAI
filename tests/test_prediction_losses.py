"""Tests for prediction/losses.py."""
import pytest
import torch

from prediction.labelers import FALL, LABEL_TO_INT, NON_FALL, PRE_IMPACT
from prediction.losses import FocalLoss, default_alpha_weights


def test_focal_loss_reduces_to_cross_entropy_when_gamma_zero_and_alpha_none():
    torch.manual_seed(0)
    logits = torch.randn(8, 3)
    targets = torch.randint(0, 3, (8,))

    focal = FocalLoss(gamma=0.0, reduction="mean")
    ce = torch.nn.CrossEntropyLoss()

    torch.testing.assert_close(focal(logits, targets), ce(logits, targets))


def test_focal_loss_downweights_confident_correct_predictions():
    # A very confident, correct prediction (huge positive logit on the
    # true class) should get a much smaller focal loss than a
    # near-random one, and the GAP between them should be larger than
    # for plain cross-entropy -- that's the whole point of the
    # (1-pt)^gamma modulating term.
    confident_logits = torch.tensor([[10.0, 0.0, 0.0]])
    unsure_logits = torch.tensor([[0.4, 0.3, 0.3]])
    target = torch.tensor([0])

    focal = FocalLoss(gamma=2.0, reduction="none")
    ce = torch.nn.CrossEntropyLoss(reduction="none")

    focal_confident, focal_unsure = focal(confident_logits, target), focal(unsure_logits, target)
    ce_confident, ce_unsure = ce(confident_logits, target), ce(unsure_logits, target)

    focal_ratio = (focal_unsure / focal_confident).item()
    ce_ratio = (ce_unsure / ce_confident).item()
    assert focal_ratio > ce_ratio


def test_focal_loss_alpha_weighting_scales_per_class():
    logits = torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    targets = torch.tensor([0, 1])  # same logits, different true class
    alpha = torch.tensor([1.0, 5.0, 1.0])  # class 1 weighted 5x

    loss = FocalLoss(alpha=alpha, gamma=2.0, reduction="none")(logits, targets)

    # Same logits/prediction difficulty for both samples (by symmetry
    # of how far the wrong prediction is from being right isn't quite
    # symmetric here, but alpha scaling is exactly linear on top of the
    # unweighted per-sample loss) -- check the ratio matches alpha's ratio
    # once compared against the unweighted version.
    unweighted = FocalLoss(alpha=None, gamma=2.0, reduction="none")(logits, targets)
    torch.testing.assert_close(loss, unweighted * alpha[targets])


def test_focal_loss_reduction_modes():
    logits = torch.randn(5, 3)
    targets = torch.randint(0, 3, (5,))

    per_sample = FocalLoss(reduction="none")(logits, targets)
    mean_loss = FocalLoss(reduction="mean")(logits, targets)
    sum_loss = FocalLoss(reduction="sum")(logits, targets)

    assert per_sample.shape == (5,)
    torch.testing.assert_close(mean_loss, per_sample.mean())
    torch.testing.assert_close(sum_loss, per_sample.sum())


def test_focal_loss_invalid_reduction_raises():
    with pytest.raises(ValueError, match="reduction"):
        FocalLoss(reduction="bogus")


# --- default_alpha_weights ---

def test_default_alpha_weights_rarer_class_gets_higher_weight():
    counts = {NON_FALL: 1000, PRE_IMPACT: 100, FALL: 300}

    weights = default_alpha_weights(counts, pre_impact_extra_boost=1.0)  # isolate inverse-freq effect only

    non_fall_w = weights[LABEL_TO_INT[NON_FALL]]
    pre_impact_w = weights[LABEL_TO_INT[PRE_IMPACT]]
    fall_w = weights[LABEL_TO_INT[FALL]]
    assert pre_impact_w > fall_w > non_fall_w


def test_default_alpha_weights_pre_impact_boost_applied():
    counts = {NON_FALL: 1000, PRE_IMPACT: 100, FALL: 300}

    weights_no_boost = default_alpha_weights(counts, pre_impact_extra_boost=1.0)
    weights_boosted = default_alpha_weights(counts, pre_impact_extra_boost=2.0)

    # NOTE: the ratio isn't exactly 2.0x, because normalization (sum
    # to n_classes) happens AFTER the boost -- boosting pre_impact's
    # raw weight also raises the total the other classes get divided
    # by, so every class's normalized weight shifts a little, not just
    # pre_impact's. Compute the expected value the same way the
    # function does, rather than assuming the boost survives
    # normalization unchanged.
    inv_freq = {NON_FALL: 1 / 1000, PRE_IMPACT: (1 / 100) * 2.0, FALL: 1 / 300}
    total = sum(inv_freq.values())
    expected_pre_impact = inv_freq[PRE_IMPACT] * (3 / total)

    assert weights_boosted[LABEL_TO_INT[PRE_IMPACT]].item() == pytest.approx(expected_pre_impact, rel=1e-4)
    # Still directionally true regardless of the exact renormalized
    # value: more boost -> strictly higher pre_impact weight.
    assert weights_boosted[LABEL_TO_INT[PRE_IMPACT]] > weights_no_boost[LABEL_TO_INT[PRE_IMPACT]]


def test_default_alpha_weights_sum_to_n_classes():
    counts = {NON_FALL: 1000, PRE_IMPACT: 100, FALL: 300}
    weights = default_alpha_weights(counts)
    assert weights.sum().item() == pytest.approx(3.0, rel=1e-4)


def test_default_alpha_weights_missing_class_raises():
    counts = {NON_FALL: 1000, FALL: 300}  # pre_impact missing
    with pytest.raises(ValueError, match="pre_impact"):
        default_alpha_weights(counts)


def test_default_alpha_weights_zero_count_raises():
    counts = {NON_FALL: 1000, PRE_IMPACT: 0, FALL: 300}
    with pytest.raises(ValueError, match="pre_impact"):
        default_alpha_weights(counts)
