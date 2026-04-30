"""Network output container for MuZero model inferences."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class NetworkOutput:
    """Output from a MuZero network's initial or recurrent inference.

    All tensors are on the same device as the input batch.

    Attributes:
        value: Predicted value, shape ``[B]`` (single) or ``[B, num_players]`` (multi).
        reward: Predicted reward, shape ``[B]`` or ``[B, num_players]``.
        policy_logits: Policy logits, shape ``[B, action_dim]``.
        hidden_state: Hidden state for the next recurrent step, shape ``[B, hidden_size]``.
        value_prefix: EfficientZero value prefix, shape ``[B, ...]`` or None.
        projection: EfficientZero projection vector, shape ``[B, projection_size]`` or None.
    """

    value: torch.Tensor
    reward: torch.Tensor
    policy_logits: torch.Tensor
    hidden_state: torch.Tensor
    value_prefix: torch.Tensor | None = None
    projection: torch.Tensor | None = None
