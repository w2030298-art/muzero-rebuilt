"""Tests for the support transform (scalar <-> categorical conversion)."""

from __future__ import annotations

import torch

from muzero.core.support import SupportTransform


def test_scalar_to_support_shape() -> None:
    """Verify scalar_to_support produces correct output shape."""
    transform = SupportTransform(support_size=10)
    x = torch.tensor([0.0, 5.0, -3.0])
    result = transform.scalar_to_support(x)
    assert result.shape == (3, 21)  # 2 * 10 + 1 = 21 bins


def test_scalar_to_support_shape_batch() -> None:
    """Verify scalar_to_support works with 2D batch input."""
    transform = SupportTransform(support_size=5)
    x = torch.randn(4, 3)
    result = transform.scalar_to_support(x)
    assert result.shape == (4, 3, 11)  # 2 * 5 + 1 = 11 bins


def test_support_to_scalar_roundtrip_for_integer_values() -> None:
    """Verify support_to_scalar approximately recovers integer values."""
    transform = SupportTransform(support_size=10)
    for val in [-5, 0, 3, 7]:
        x = torch.tensor([float(val)])
        support_logits = transform.scalar_to_support(x)
        # Converting support back should be close to original
        recovered = transform.support_to_scalar(support_logits * 100)
        assert abs(float(recovered) - val) < 1.0


def test_scalar_to_support_clamps_out_of_range() -> None:
    """Verify values outside support range are clamped."""
    transform = SupportTransform(support_size=10)
    x = torch.tensor([-20.0, 0.0, 30.0])
    result = transform.scalar_to_support(x)
    # All rows should sum to 1 (valid probability distribution)
    sums = result.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_scalar_to_support_precision() -> None:
    """Verify scalar_to_support places mass on nearest bins."""
    transform = SupportTransform(support_size=10)
    # At value 0, the center of bin at index support_size should have highest weight
    x = torch.tensor([0.0])
    result = transform.scalar_to_support(x)
    # Middle bin (index 10 for support_size=10) should have the highest weight
    max_idx = result.argmax().item()
    assert max_idx == 10  # center of 21 bins


def test_support_to_scalar_batch() -> None:
    """Verify support_to_scalar works with batched logits."""
    transform = SupportTransform(support_size=5)
    logits = torch.randn(3, 11)  # 3 samples, 11 bins
    result = transform.support_to_scalar(logits)
    assert result.shape == (3,)
    # Values should be within support range
    assert float(result.min()) >= -5.0
    assert float(result.max()) <= 5.0


def test_support_property() -> None:
    """Verify the support property returns correct bin centers."""
    transform = SupportTransform(support_size=3)
    support = transform.support
    assert support.shape == (7,)  # 2*3+1
    assert float(support[0]) == -3.0
    assert float(support[-1]) == 3.0
    assert float(support[3]) == 0.0


def test_invalid_support_size() -> None:
    """Verify that support_size < 1 raises ValueError."""
    import pytest

    with pytest.raises(ValueError):
        SupportTransform(support_size=0)
