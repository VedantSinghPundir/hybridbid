"""Tests for Transformer Temporal Feature Extractor."""

import torch
import pytest
from src.models.ttfe import TTFE


@pytest.fixture
def ttfe():
    return TTFE(n_prices=12, d_model=64, nhead=4, n_layers=2, seq_len=32)


def test_output_shape(ttfe):
    x = torch.randn(8, 32, 12)
    out = ttfe(x)
    assert out.shape == (8, 64)


def test_gradient_flow(ttfe):
    x = torch.randn(4, 32, 12)
    out = ttfe(x)
    loss = out.sum()
    loss.backward()
    # All parameters should have gradients
    for name, param in ttfe.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_zeros_in_rt_mcpc_channels(ttfe):
    """Simulate pre-RTC+B data where RT MCPC channels (indices 1-5) are zero."""
    x = torch.randn(4, 32, 12)
    x[:, :, 1:6] = 0.0  # RT MCPC channels zeroed
    out = ttfe(x)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()


def test_different_batch_sizes(ttfe):
    for batch_size in [1, 4, 16, 32]:
        x = torch.randn(batch_size, 32, 12)
        out = ttfe(x)
        assert out.shape == (batch_size, 64)


def test_single_sample(ttfe):
    x = torch.randn(1, 32, 12)
    out = ttfe(x)
    assert out.shape == (1, 64)
