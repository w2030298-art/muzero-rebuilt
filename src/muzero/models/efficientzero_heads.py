"""EfficientZero auxiliary heads: value prefix and consistency projection.

These heads are attached to MLPNetwork when ``use_value_prefix`` or
``use_consistency_loss`` is enabled in the algorithm config.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientZeroHeads(nn.Module):
    """Auxiliary heads for EfficientZero-style training.

    Args:
        hidden_size: Dimension of the hidden state input.
        projection_size: Dimension of the projection/consistency vector.
    """

    def __init__(self, hidden_size: int, projection_size: int = 128) -> None:
        super().__init__()
        self._hidden_size = hidden_size
        self._projection_size = projection_size

        # Value prefix head: predict cumulative future reward
        self._value_prefix_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # Projection head: encode hidden state for consistency
        self._projection_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_size),
        )

        # Prediction projection: predict projection from dynamics output
        self._prediction_projection_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_size),
        )

    def value_prefix(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Predict value prefix (cumulative future reward from dynamics).

        Args:
            hidden_state: Hidden state tensor, shape ``[B, hidden_size]``.

        Returns:
            Value prefix, shape ``[B, 1]``.
        """
        return self._value_prefix_net(hidden_state)

    def projection(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project hidden state to a normalized vector for consistency loss.

        The output is L2-normalized along the last dimension.

        Args:
            hidden_state: Hidden state tensor, shape ``[B, hidden_size]``.

        Returns:
            L2-normalized projection vector, shape ``[B, projection_size]``.
        """
        proj = self._projection_net(hidden_state)
        return F.normalize(proj, p=2, dim=-1)

    def prediction_projection(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Predict projection from dynamics output for consistency target.

        The output is L2-normalized along the last dimension.

        Args:
            hidden_state: Hidden state tensor, shape ``[B, hidden_size]``.

        Returns:
            L2-normalized prediction vector, shape ``[B, projection_size]``.
        """
        pred = self._prediction_projection_net(hidden_state)
        return F.normalize(pred, p=2, dim=-1)
