"""Tests for prediction/models/ -- ConvLSTM and TinyTransformer forward
pass correctness (shape, dtype, gradient flow). Not accuracy tests --
these models are untrained; the point is to confirm the architecture
is wired correctly, matches the Dataset's output layout, and is
actually trainable (gradients reach every parameter)."""
import torch

from prediction.models.convlstm import N_CLASSES as CONVLSTM_N_CLASSES
from prediction.models.convlstm import N_INPUT_CHANNELS as CONVLSTM_N_INPUT_CHANNELS
from prediction.models.convlstm import ConvLSTM
from prediction.models.tiny_transformer import SEQ_LEN
from prediction.models.tiny_transformer import N_CLASSES as TRANSFORMER_N_CLASSES
from prediction.models.tiny_transformer import N_INPUT_CHANNELS as TRANSFORMER_N_INPUT_CHANNELS
from prediction.models.tiny_transformer import TinyTransformer


def _dummy_batch(batch_size=4, channels=9, seq_len=100):
    return torch.randn(batch_size, channels, seq_len)


# --- ConvLSTM ---

def test_convlstm_output_shape():
    model = ConvLSTM()
    x = _dummy_batch()

    logits = model(x)

    assert logits.shape == (4, CONVLSTM_N_CLASSES)


def test_convlstm_defaults_match_dataset_output_layout():
    assert CONVLSTM_N_INPUT_CHANNELS == 9  # matches prediction.features.AUX_CHANNEL_NAMES + CHANNELS = 9
    assert CONVLSTM_N_CLASSES == 3


def test_convlstm_gradients_reach_all_parameters():
    model = ConvLSTM()
    x = _dummy_batch()
    logits = model(x)
    logits.sum().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"


def test_convlstm_handles_batch_size_one():
    # BatchNorm1d in the conv stack can break on batch_size=1 during
    # training mode (needs >1 sample to compute batch statistics) --
    # confirm eval mode at least handles it, since a real training loop
    # may hit a trailing partial batch.
    model = ConvLSTM()
    model.eval()
    x = _dummy_batch(batch_size=1)

    logits = model(x)

    assert logits.shape == (1, CONVLSTM_N_CLASSES)


# --- TinyTransformer ---

def test_tiny_transformer_output_shape():
    model = TinyTransformer()
    x = _dummy_batch()

    logits = model(x)

    assert logits.shape == (4, TRANSFORMER_N_CLASSES)


def test_tiny_transformer_defaults_match_dataset_output_layout():
    assert TRANSFORMER_N_INPUT_CHANNELS == 9
    assert TRANSFORMER_N_CLASSES == 3
    assert SEQ_LEN == 100  # matches PredictionWindowingConfig's window_length_samples


def test_tiny_transformer_gradients_reach_all_parameters():
    model = TinyTransformer()
    x = _dummy_batch()
    logits = model(x)
    logits.sum().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"


def test_tiny_transformer_handles_batch_size_one():
    model = TinyTransformer()
    x = _dummy_batch(batch_size=1)

    logits = model(x)

    assert logits.shape == (1, TRANSFORMER_N_CLASSES)


def test_tiny_transformer_rejects_d_model_not_divisible_by_n_heads():
    import pytest
    with pytest.raises(ValueError, match="divisible"):
        TinyTransformer(d_model=50, n_heads=3)


# --- Both branches share the same input/output contract ---

def test_both_branches_accept_identical_input_and_produce_identical_output_shape():
    x = _dummy_batch()
    convlstm_logits = ConvLSTM()(x)
    transformer_logits = TinyTransformer()(x)

    assert convlstm_logits.shape == transformer_logits.shape == (4, 3)
