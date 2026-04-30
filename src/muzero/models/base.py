"""Base MuZero network with representation/dynamics/prediction structure.

Provides default implementations of ``initial_inference`` and ``recurrent_inference``
that delegate to the abstract ``representation``, ``dynamics``, and ``prediction`` methods.
Subclasses only need to implement these three methods.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from muzero.models.outputs import NetworkOutput


class BaseMuZeroNetwork(nn.Module):
    """Abstract base class for MuZero network architectures.

    Subclasses must implement:
        - ``representation(observation) → hidden_state``
        - ``dynamics(hidden_state, action) → (next_hidden_state, reward)``
        - ``prediction(hidden_state) → (policy_logits, value)``

    The ``initial_inference`` and ``recurrent_inference`` methods are provided
    and should not be overridden unless absolutely necessary.
    """

    def representation(self, observation_batch: torch.Tensor) -> torch.Tensor:
        """Encode an observation into a hidden state.

        Args:
            observation_batch: Batch of observations, shape ``[B, *obs_shape]``.

        Returns:
            Hidden state tensor, shape ``[B, hidden_size]``.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement representation()")

    def dynamics(
        self, hidden_state_batch: torch.Tensor, action_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the next hidden state and reward for a given action.

        Args:
            hidden_state_batch: Current hidden states, shape ``[B, hidden_size]``.
            action_batch: Actions to apply, shape ``[B, ...]``.

        Returns:
            Tuple of ``(next_hidden_state, reward)``.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement dynamics()")

    def prediction(self, hidden_state_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict policy logits and value from a hidden state.

        Args:
            hidden_state_batch: Hidden states, shape ``[B, hidden_size]``.

        Returns:
            Tuple of ``(policy_logits, value)``.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement prediction()")

    def initial_inference(self, observation_batch: torch.Tensor) -> NetworkOutput:
        """Perform initial inference from observations.

        Args:
            observation_batch: Batch of observations, shape ``[B, *obs_shape]``.

        Returns:
            NetworkOutput with value, reward=0, policy_logits, and hidden_state.
        """
        hidden_state = self.representation(observation_batch)
        policy_logits, value = self.prediction(hidden_state)

        # Initial reward is zero
        reward = torch.zeros(
            value.shape[0],
            *value.shape[1:],
            dtype=value.dtype,
            device=value.device,
        )

        return NetworkOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=hidden_state,
        )

    def recurrent_inference(
        self, hidden_state_batch: torch.Tensor, action_batch: torch.Tensor
    ) -> NetworkOutput:
        """Perform recurrent inference: dynamics → prediction.

        Args:
            hidden_state_batch: Hidden states from previous step, shape ``[B, hidden_size]``.
            action_batch: Actions taken, shape ``[B, ...]``.

        Returns:
            NetworkOutput with value, reward, policy_logits, and next hidden_state.
        """
        next_hidden_state, reward = self.dynamics(hidden_state_batch, action_batch)
        policy_logits, value = self.prediction(next_hidden_state)

        return NetworkOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=next_hidden_state,
        )

    def get_weights(self) -> dict[str, torch.Tensor]:
        """Return a CPU copy of all model parameters.

        Returns:
            Dictionary mapping parameter names to detached CPU tensor copies.
        """
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Load weights into the model using ``load_state_dict``.

        Args:
            weights: Dictionary mapping parameter names to tensors.
        """
        self.load_state_dict(weights)
