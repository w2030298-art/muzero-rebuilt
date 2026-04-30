"""Self-play worker for generating game episodes."""

from __future__ import annotations

from muzero.config.schema import MuZeroConfig
from muzero.core.game_history import GameHistory
from muzero.core.types import Action, SearchResult
from muzero.envs.base import GameAdapter
from muzero.search.mcts import MCTS


class SelfPlayWorker:
    """Runs self-play episodes against the environment using MCTS.

    Args:
        env: GameAdapter for the environment.
        search: MCTS instance for action selection.
        config: Full MuZero configuration.
    """

    def __init__(
        self,
        env: GameAdapter,
        search: MCTS,
        config: MuZeroConfig,
    ) -> None:
        self._env = env
        self._search = search
        self._config = config
        max_steps = config.environment.max_episode_steps
        self._max_steps = max_steps if max_steps is not None else 500

    def run_episode(self, seed: int | None = None) -> GameHistory:
        """Run one full self-play episode.

        Args:
            seed: Optional random seed.

        Returns:
            GameHistory with the full episode trajectory.
        """
        history = GameHistory()
        timestep = self._env.reset(seed=seed)
        history.append_initial(timestep)

        steps = 0
        while not timestep.done and steps < self._max_steps:
            # Run MCTS search
            search_result = self._search.run(
                root_observation=timestep.observation,
                legal_actions=timestep.legal_actions,
                to_play=timestep.to_play,
            )

            # Select action
            action = self._select_action(search_result, self._config.search.temperature)

            # Step the environment
            timestep = self._env.step(action)
            history.append(action=action, timestep=timestep, search_result=search_result)
            steps += 1

        return history

    def _select_action(self, search_result: SearchResult, temperature: float) -> Action:
        """Select action from search result with temperature.

        Args:
            search_result: MCTS search result.
            temperature: Temperature for action selection.

        Returns:
            Selected action.
        """
        return search_result.action
