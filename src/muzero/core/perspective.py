"""Player perspective transformation for multi-player games.

Projects multi-player value vectors to a single-player perspective,
supporting 1-player, 2-player zero-sum, and N-player general-sum games.
"""

from __future__ import annotations

import torch


class PlayerPerspective:
    """Transforms value/reward vectors to the perspective of a specific player.

    Rules:
        - ``num_players == 1``: Identity (returns original value).
        - If the last dimension of ``value`` equals ``num_players``, treat as a player
          vector and project to the current player's perspective.
        - If the last dimension differs, return value unchanged (treated as scalar batch).
        - ``num_players == 2`` with vector input: ``value[..., to_play] - value[..., 1-to_play]``.
        - ``num_players > 2`` with vector input: ``value[..., to_play] - mean(value[..., others])``.
    """

    @staticmethod
    def project_value(value: torch.Tensor, to_play: int, num_players: int) -> torch.Tensor:
        """Project a value vector to the perspective of the current player.

        Args:
            value: Value tensor. May be scalar ``[B]`` or vector ``[B, num_players]``.
            to_play: Index of the current player (0-based).
            num_players: Total number of players.

        Returns:
            Projected scalar value for the current player, shape ``[B]``.
        """
        if num_players == 1:
            if value.dim() > 1 and value.shape[-1] == 1:
                return value.squeeze(-1)
            return value

        # Determine if this is a player vector (last dim == num_players)
        is_player_vector = value.dim() >= 1 and value.shape[-1] == num_players

        if not is_player_vector:
            return value

        if num_players == 2:
            return value[..., to_play] - value[..., 1 - to_play]

        # num_players > 2: value[to_play] - mean(value[others])
        others_mask = torch.ones(num_players, dtype=torch.bool, device=value.device)
        others_mask[to_play] = False
        mean_others = value[..., others_mask].mean(dim=-1)
        return value[..., to_play] - mean_others

    @staticmethod
    def project_reward(reward: torch.Tensor, to_play: int, num_players: int) -> torch.Tensor:
        """Project a reward vector to the perspective of the current player.

        Uses the same rules as :meth:`project_value`.

        Args:
            reward: Reward tensor. May be scalar or vector.
            to_play: Index of the current player (0-based).
            num_players: Total number of players.

        Returns:
            Projected scalar reward for the current player.
        """
        return PlayerPerspective.project_value(reward, to_play, num_players)
