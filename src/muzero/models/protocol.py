"""Protocol defining the MuZero network interface.

All network implementations must satisfy this protocol so that
the search, training, and execution layers can use them interchangeably.
"""

from __future__ import annotations

from typing import Protocol

import torch

from muzero.models.outputs import NetworkOutput


class MuZeroNetworkProtocol(Protocol):
    """Unified protocol for all MuZero network variants.

    The two main entry points are ``initial_inference`` and ``recurrent_inference``.
    These are the ONLY methods the search/execution layers should call.
    Internal methods like ``representation``, ``dynamics``, ``prediction``
    are implementation details of ``BaseMuZeroNetwork``.
    """

    def initial_inference(self, observation_batch: torch.Tensor) -> NetworkOutput:
        """Perform the initial inference from an observation.

        Calls ``representation(obs) → hidden_state`` followed by
        ``prediction(hidden_state) → (policy_logits, value)``.
        The reward is set to a zero tensor.

        Args:
            observation_batch: Batch of observations, shape ``[B, *obs_shape]``.

        Returns:
            NetworkOutput with value, reward (zeros), policy_logits, and hidden_state.
        """
        ...

    def recurrent_inference(
        self, hidden_state_batch: torch.Tensor, action_batch: torch.Tensor
    ) -> NetworkOutput:
        """Perform inference for a recurrent step.

        Calls ``dynamics(hidden_state, action) → (next_hidden_state, reward)``
        followed by ``prediction(next_hidden_state) → (policy_logits, value)``.

        Args:
            hidden_state_batch: Hidden states from the previous step, shape ``[B, hidden_size]``.
            action_batch: Actions taken, shape ``[B, ...]``.

        Returns:
            NetworkOutput with value, reward, policy_logits, and next hidden_state.
        """
        ...

    def get_weights(self) -> dict[str, torch.Tensor]:
        """Return a copy of the model's weights as CPU tensors.

        Returns:
            Dictionary mapping parameter names to CPU tensor copies.
        """
        ...

    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Load weights into the model.

        Args:
            weights: Dictionary mapping parameter names to tensors.
        """
        ...
