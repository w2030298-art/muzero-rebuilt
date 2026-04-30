"""
Tests for the loss functions with synthetic data.
"""

from __future__ import annotations

import torch

from muzero.training.losses import MuZeroLoss


def test_policy_loss() -> None:
    loss_fn = MuZeroLoss()
    pred = torch.randn(4, 5)
    target = torch.softmax(torch.randn(4, 5), dim=-1)
    result = loss_fn.policy_loss(pred, target)
    assert result.ndim == 0
    assert result.item() > 0


def test_value_loss() -> None:
    loss_fn = MuZeroLoss()
    pred = torch.randn(4)
    target = torch.randn(4)
    result = loss_fn.value_loss(pred, target)
    assert result.ndim == 0
    assert result.item() >= 0


def test_reward_loss() -> None:
    loss_fn = MuZeroLoss()
    pred = torch.randn(4)
    target = torch.randn(4)
    result = loss_fn.reward_loss(pred, target)
    assert result.ndim == 0
    assert result.item() >= 0
