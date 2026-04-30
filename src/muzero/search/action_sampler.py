"""Action samplers for MCTS tree expansion.

For discrete action spaces, samples top-k actions by prior probability.
For continuous action spaces, see ``ContinuousActionSampler`` (Module 13).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch


@dataclass(slots=True)
class ActionSampleBatch:
    """Batch of sampled actions with their priors.

    Attributes:
        actions: Sampled action indices or vectors, shape ``[B, K]`` or ``[B, K, action_dim]``.
        priors: Prior probabilities for each sampled action, shape ``[B, K]``.
        log_probs: Optional log probabilities, shape ``[B, K]`` (set for continuous).
    """

    actions: torch.Tensor
    priors: torch.Tensor
    log_probs: torch.Tensor | None = None


class ActionSampler(Protocol):
    """Protocol for action sampling strategies.

    The sampler decides which subset of actions to explore in the MCTS tree.
    For discrete actions with small action spaces, all actions are sampled.
    For large discrete or continuous spaces, a subset is used.
    """

    def sample(
        self,
        policy_output: torch.Tensor,
        hidden_state: torch.Tensor,
        num_samples: int,
        legal_action_mask: torch.Tensor | None,
    ) -> ActionSampleBatch:
        """Sample a subset of actions to explore.

        Args:
            policy_output: Raw policy logits from the network, shape ``[B, action_dim]``.
            hidden_state: Hidden states from the network, shape ``[B, hidden_size]``.
            num_samples: Number of actions to sample per batch element.
            legal_action_mask: Optional boolean mask of legal actions, shape ``[B, action_dim]``.

        Returns:
            ActionSampleBatch with sampled actions and their priors.
        """
        ...


class DiscreteActionSampler:
    """Samples discrete actions based on policy probabilities.

    If ``num_samples >= action_dim``, all actions are returned.
    Otherwise, the top-k actions by prior probability are selected.
    Illegal actions (according to ``legal_action_mask``) are excluded.
    """

    def sample(
        self,
        policy_output: torch.Tensor,
        hidden_state: torch.Tensor,
        num_samples: int,
        legal_action_mask: torch.Tensor | None,
    ) -> ActionSampleBatch:
        """Sample discrete actions from the policy output.

        Args:
            policy_output: Raw policy logits, shape ``[B, action_dim]``.
            hidden_state: Hidden states (unused by this sampler).
            num_samples: Number of actions to sample per batch element.
            legal_action_mask: Legal action mask, shape ``[B, action_dim]`` or None.

        Returns:
            ActionSampleBatch with actions shape ``[B, K]`` and priors shape ``[B, K]``.
        """
        batch_size, action_dim = policy_output.shape
        logits = policy_output

        # Apply legal action mask: set illegal logits to -inf
        if legal_action_mask is not None:
            logits = logits.masked_fill(~legal_action_mask, float("-inf"))

        probs = torch.softmax(logits, dim=-1)

        k = min(num_samples, action_dim)

        if k >= action_dim:
            # Return all actions
            actions = torch.arange(action_dim, device=policy_output.device)
            actions = actions.unsqueeze(0).expand(batch_size, -1)
            priors = probs
        else:
            # Top-k actions by probability
            topk_priors, topk_indices = torch.topk(probs, k, dim=-1)
            actions = topk_indices
            priors = topk_priors
            # Re-normalize priors
            priors = priors / priors.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return ActionSampleBatch(actions=actions, priors=priors)


class ContinuousActionSampler:
    """Samples continuous actions from a Gaussian around the policy mean.

    For Sampled MuZero: policy output is interpreted as action mean,
    and K actions are sampled from Normal(mean, std). Priors are uniform.

    Args:
        action_low: Lower bounds for each action dimension.
        action_high: Upper bounds for each action dimension.
        sampling_std: Standard deviation for action sampling.
    """

    def __init__(
        self,
        action_low: np.ndarray,  # noqa: ARG002
        action_high: np.ndarray,  # noqa: ARG002
        sampling_std: float = 0.3,
    ) -> None:
        self._low = torch.from_numpy(action_low).float()
        self._high = torch.from_numpy(action_high).float()
        self._std = sampling_std

    def sample(
        self,
        policy_output: torch.Tensor,
        hidden_state: torch.Tensor,  # noqa: ARG002
        num_samples: int,
        legal_action_mask: torch.Tensor | None,  # noqa: ARG002
    ) -> ActionSampleBatch:
        """Sample continuous actions around the policy mean.

        Args:
            policy_output: Action means from network, shape ``[B, action_dim]``.
            hidden_state: Hidden states (unused).
            num_samples: Number of actions to sample per batch element.
            legal_action_mask: Unused for continuous actions.

        Returns:
            ActionSampleBatch with actions shape ``[B, K, action_dim]``
            and uniform priors shape ``[B, K]``.
        """
        B, _ = policy_output.shape
        self._low = self._low.to(policy_output.device)
        self._high = self._high.to(policy_output.device)

        # Expand means: [B, 1, action_dim] → [B, K, action_dim]
        means = policy_output.unsqueeze(1).expand(-1, num_samples, -1)

        # Sample from Gaussian
        actions = means + self._std * torch.randn_like(means)

        # Clamp to bounds
        actions = torch.clamp(actions, self._low, self._high)

        # Uniform priors
        priors = torch.ones(B, num_samples, device=policy_output.device) / num_samples

        return ActionSampleBatch(actions=actions, priors=priors)
