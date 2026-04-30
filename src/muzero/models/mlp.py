"""MLP-based MuZero network for low-dimensional observation spaces.

Suitable for CartPole, LunarLander, and other environments with
flat vector observations and discrete action spaces.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from muzero.models.base import BaseMuZeroNetwork
from muzero.models.outputs import NetworkOutput


class MLPNetwork(BaseMuZeroNetwork):
    """MLP-based MuZero network.

    Architecture:
        - ``representation_net``: Flatten → Linear(h+w) → ReLU → Linear(h+h) → ReLU
        - ``dynamics_state_net``: Linear(h + action_dim, h) → ReLU → Linear(h, h) → ReLU
        - ``reward_head``: Linear(h, h) → ReLU → Linear(h, n_val)
        - ``policy_head``: Linear(h, h) → ReLU → Linear(h, action_dim)
        - ``value_head``: Linear(h, h) → ReLU → Linear(h, n_val)

    Where ``n_val`` = 1 (single-player) or ``num_players`` (multi-player).

    Args:
        observation_shape: Shape of flattened observation input.
        action_space_size: Number of discrete actions.
        hidden_size: Hidden layer size.
        num_players: Number of players (1 = single, 2+ = multi).
        use_value_prefix: Whether to include EfficientZero value prefix head.
        use_consistency_loss: Whether to include EfficientZero projection head.
    """

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_space_size: int,
        hidden_size: int = 128,
        num_players: int = 1,
        use_value_prefix: bool = False,
        use_consistency_loss: bool = False,
        action_space_type: str = "discrete",
        action_shape: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()

        self._hidden_size = hidden_size
        self._num_players = num_players
        self._action_space_size = action_space_size
        self._action_space_type = action_space_type
        self._use_value_prefix = use_value_prefix
        self._use_consistency_loss = use_consistency_loss

        n_val = 1 if num_players == 1 else num_players

        # Compute observation dimension
        obs_dim = 1
        for s in observation_shape:
            obs_dim *= s

        # Compute action input dimension
        if action_space_type == "discrete":
            action_input_dim = action_space_size
        else:
            action_input_dim = 1
            if action_shape is not None:
                for s in action_shape:
                    action_input_dim *= s

        # Representation network
        self.representation_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Dynamics network
        self.dynamics_state_net = nn.Sequential(
            nn.Linear(hidden_size + action_input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_val),
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space_size),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_val),
        )

        # EfficientZero heads (optional)
        self._ez_heads: nn.Module | None = None
        if use_value_prefix or use_consistency_loss:
            from muzero.models.efficientzero_heads import EfficientZeroHeads

            self._ez_heads = EfficientZeroHeads(hidden_size)

    def representation(self, observation_batch: torch.Tensor) -> torch.Tensor:
        """Encode observation into hidden state.

        Args:
            observation_batch: Observations, shape ``[B, *obs_shape]``.

        Returns:
            Hidden state, shape ``[B, hidden_size]``.
        """
        return self.representation_net(observation_batch)

    def dynamics(
        self, hidden_state_batch: torch.Tensor, action_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute next hidden state and reward.

        For discrete actions, encodes action as one-hot before concatenation.
        For continuous actions, flattens the action tensor.

        Args:
            hidden_state_batch: Hidden states, shape ``[B, hidden_size]``.
            action_batch: Actions, shape ``[B]`` (discrete) or ``[B, action_dim]`` (continuous).

        Returns:
            Tuple of ``(next_hidden_state, reward)``.
        """
        if self._action_space_type == "discrete":
            # One-hot encode discrete actions
            action_encoded = F.one_hot(
                action_batch.long(), num_classes=self._action_space_size
            ).float()
        else:
            # Flatten continuous actions
            action_encoded = action_batch.float()
            if action_encoded.dim() == 1:
                action_encoded = action_encoded.unsqueeze(-1)

        x = torch.cat([hidden_state_batch, action_encoded], dim=-1)
        next_hidden_state = self.dynamics_state_net(x)
        reward = self.reward_head(next_hidden_state)

        # Squeeze single-player reward from [B,1] to [B]
        if self._num_players == 1 and reward.shape[-1] == 1:
            reward = reward.squeeze(-1)

        return next_hidden_state, reward

    def prediction(self, hidden_state_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict policy logits and value from hidden state.

        Args:
            hidden_state_batch: Hidden states, shape ``[B, hidden_size]``.

        Returns:
            Tuple of ``(policy_logits, value)``.
        """
        policy_logits = self.policy_head(hidden_state_batch)
        value = self.value_head(hidden_state_batch)

        # Squeeze single-player value from [B,1] to [B]
        if self._num_players == 1 and value.shape[-1] == 1:
            value = value.squeeze(-1)

        return policy_logits, value

    def initial_inference(self, observation_batch: torch.Tensor) -> NetworkOutput:
        """Override to add EfficientZero output fields if enabled.

        Args:
            observation_batch: Batch of observations.

        Returns:
            NetworkOutput with optional value_prefix and projection.
        """
        output = super().initial_inference(observation_batch)

        if self._ez_heads is not None:
            ez: nn.Module = self._ez_heads
            if self._use_value_prefix:
                output.value_prefix = ez.value_prefix(output.hidden_state)  # type: ignore[reportCallIssue]
            if self._use_consistency_loss:
                output.projection = ez.projection(output.hidden_state)  # type: ignore[reportCallIssue]

        return output

    def recurrent_inference(
        self, hidden_state_batch: torch.Tensor, action_batch: torch.Tensor
    ) -> NetworkOutput:
        """Override to add EfficientZero output fields if enabled.

        Args:
            hidden_state_batch: Hidden states.
            action_batch: Actions.

        Returns:
            NetworkOutput with optional value_prefix and projection.
        """
        output = super().recurrent_inference(hidden_state_batch, action_batch)

        if self._ez_heads is not None:
            ez: nn.Module = self._ez_heads
            if self._use_value_prefix:
                output.value_prefix = ez.value_prefix(output.hidden_state)  # type: ignore[reportCallIssue]
            if self._use_consistency_loss:
                output.projection = ez.projection(output.hidden_state)  # type: ignore[reportCallIssue]

        return output
