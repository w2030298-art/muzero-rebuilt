"""Gymnasium environment adapter.

Converts Gymnasium environments to the unified ``GameAdapter`` interface.
Preserves ``terminated`` / ``truncated`` distinction from Gymnasium API.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from muzero.core.specs import ActionSpaceSpec, ObservationSpaceSpec
from muzero.core.types import Action, LegalActions, TimeStep


class GymnasiumAdapter:
    """Wraps a Gymnasium environment as a ``GameAdapter``.

    For single-agent Gymnasium environments, ``current_player()`` returns 0
    and ``num_players()`` returns 1. There is no legal action mask.

    Args:
        env_id: Gymnasium environment ID (e.g., ``"CartPole-v1"``).
        max_episode_steps: Optional override for max episode steps.
    """

    def __init__(self, env_id: str, max_episode_steps: int | None = None) -> None:
        self._env_id = env_id
        kwargs: dict[str, Any] = {}
        if max_episode_steps is not None:
            kwargs["max_episode_steps"] = max_episode_steps
        self._env = gym.make(env_id, **kwargs)
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space

    def reset(self, seed: int | None = None) -> TimeStep:
        """Reset the environment and return the initial observation.

        Args:
            seed: Optional random seed.

        Returns:
            Initial TimeStep with reward=0, terminated=False, truncated=False.
        """
        obs, info = self._env.reset(seed=seed)
        return TimeStep(
            observation=np.asarray(obs, dtype=np.float32),
            reward=0.0,
            terminated=False,
            truncated=False,
            to_play=0,
            info=info,
        )

    def step(self, action: Action) -> TimeStep:
        """Execute an action in the environment.

        Args:
            action: The action to take.

        Returns:
            TimeStep with observation, reward, terminated, truncated, and to_play.
        """
        obs, reward, terminated, truncated, info = self._env.step(action)  # type: ignore[arg-type]
        return TimeStep(
            observation=np.asarray(obs, dtype=np.float32),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            to_play=0,
            info=info,
        )

    def legal_actions(self) -> LegalActions:
        """GymnasiumAdapter has no legal action mask.

        Returns:
            None (all actions are considered legal).
        """
        return None

    def current_player(self) -> int:
        """Single-agent environments always have player 0.

        Returns:
            0
        """
        return 0

    def num_players(self) -> int:
        """GymnasiumAdapter always wraps a single agent.

        Returns:
            1
        """
        return 1

    def action_space_spec(self) -> ActionSpaceSpec:
        """Return the specification of the action space.

        Returns:
            ActionSpaceSpec describing discrete or continuous action space.
        """
        if isinstance(self._action_space, gym.spaces.Discrete):
            return ActionSpaceSpec(
                type="discrete",
                n=int(self._action_space.n),
            )
        else:
            # Box action space (continuous)
            low = self._action_space.low  # type: ignore[union-attr]
            high = self._action_space.high  # type: ignore[union-attr]
            shape = self._action_space.shape  # type: ignore[union-attr]
            if shape is None:
                shape = (1,)
            if isinstance(shape, int):
                shape = (shape,)
            return ActionSpaceSpec(
                type="continuous",
                shape=tuple(shape),
                low=low.tolist() if hasattr(low, "tolist") else list(low),
                high=high.tolist() if hasattr(high, "tolist") else list(high),
            )

    def observation_space_spec(self) -> ObservationSpaceSpec:
        """Return the specification of the observation space.

        Returns:
            ObservationSpaceSpec describing shape and dtype.
        """
        shape = self._observation_space.shape
        if shape is None:
            shape = (1,)
        if isinstance(shape, int):
            shape = (shape,)
        return ObservationSpaceSpec(
            shape=tuple(shape),
            dtype=str(self._observation_space.dtype),
        )

    def render(self, mode: str = "human") -> Any:
        """Render the current environment state.

        Args:
            mode: Rendering mode.

        Returns:
            Render output.
        """
        return self._env.render()  # type: ignore[no-any-return]

    def close(self) -> None:
        """Close the Gymnasium environment."""
        self._env.close()
