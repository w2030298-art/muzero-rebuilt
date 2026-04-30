"""Three-player toy environment for multi-player value vector testing.

3 players take turns. Action space = 2 choices. Episode length = 3.
Reward: action=1 gives +1 to current player, others get 0.
"""

from __future__ import annotations

import numpy as np

from muzero.core.specs import ActionSpaceSpec, ObservationSpaceSpec
from muzero.core.types import Action, LegalActions, TimeStep


class ThreePlayerToyEnv:
    """Simple 3-player environment for structural testing of value vectors.

    Each player acts exactly once. Reward vector is based on actions.
    """

    def __init__(self) -> None:
        self._step_count = 0
        self._to_play = 0

    def reset(self, seed: int | None = None) -> TimeStep:
        """Reset to initial state."""
        if seed is not None:
            np.random.seed(seed)
        self._step_count = 0
        self._to_play = 0
        return TimeStep(
            observation=np.array([0.0], dtype=np.float32),
            reward=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            terminated=False,
            truncated=False,
            to_play=0,
        )

    def step(self, action: Action) -> TimeStep:
        """Take an action. Reward = +1 to current player if action=1."""
        a = int(action)  # type: ignore[arg-type]
        reward = np.zeros(3, dtype=np.float32)
        if a == 1:
            reward[self._to_play] = 1.0

        self._step_count += 1
        self._to_play = (self._to_play + 1) % 3

        terminated = self._step_count >= 3
        return TimeStep(
            observation=np.array([float(self._step_count)], dtype=np.float32),
            reward=reward,
            terminated=terminated,
            truncated=False,
            to_play=self._to_play,
        )

    def legal_actions(self) -> LegalActions:
        """All actions always legal."""
        return np.array([True, True])

    def current_player(self) -> int:
        return self._to_play

    def num_players(self) -> int:
        return 3

    def action_space_spec(self) -> ActionSpaceSpec:
        return ActionSpaceSpec(type="discrete", n=2)

    def observation_space_spec(self) -> ObservationSpaceSpec:
        return ObservationSpaceSpec(shape=(1,), dtype="float32")

    def render(self, mode: str = "human") -> str:  # noqa: ARG002
        return f"Step {self._step_count}, Player {self._to_play}"

    def close(self) -> None:
        pass
