"""Environment factory that creates GameAdapter instances from configuration."""

from __future__ import annotations

from collections.abc import Callable

from muzero.config.schema import EnvironmentConfig
from muzero.envs.base import GameAdapter


class EnvFactory:
    """Registry and factory for creating GameAdapter instances.

    Built-in types (gymnasium, board_game, continuous_control) are pre-registered.
    Custom environments can be added via ``register()``.
    """

    def __init__(self) -> None:
        self._registry: dict[str, Callable[[EnvironmentConfig], GameAdapter]] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in environment types."""
        self.register("gymnasium", self._make_gymnasium)
        self.register("continuous_control", self._make_gymnasium)
        self.register("board_game", self._make_board_game)

    @staticmethod
    def _make_gymnasium(config: EnvironmentConfig) -> GameAdapter:
        """Create a GymnasiumAdapter from config."""
        from muzero.envs.gymnasium_adapter import GymnasiumAdapter

        return GymnasiumAdapter(
            env_id=config.id,
            max_episode_steps=config.max_episode_steps,
        )

    @staticmethod
    def _make_board_game(config: EnvironmentConfig) -> GameAdapter:
        """Create a board game adapter from config."""
        game_id = config.id.lower()

        if game_id == "tictactoe":
            from muzero.envs.board_games.tictactoe import TicTacToeEnv

            return TicTacToeEnv()
        elif game_id == "connect4":
            from muzero.envs.board_games.connect4 import Connect4Env

            return Connect4Env()
        elif game_id == "three_player_toy":
            from muzero.envs.board_games.three_player_toy import ThreePlayerToyEnv

            return ThreePlayerToyEnv()
        else:
            raise ValueError(f"Unknown board game: {game_id}. Supported: tictactoe, connect4")

    def register(
        self,
        name: str,
        factory: Callable[[EnvironmentConfig], GameAdapter],
    ) -> None:
        """Register a custom environment factory.

        Args:
            name: Environment type key (used in config ``environment.type``).
            factory: Callable that takes EnvironmentConfig and returns a GameAdapter.
        """
        self._registry[name] = factory

    def make(self, config: EnvironmentConfig) -> GameAdapter:
        """Create a GameAdapter from the given configuration.

        Args:
            config: Environment configuration specifying type, id, etc.

        Returns:
            A GameAdapter instance for the specified environment.

        Raises:
            KeyError: If the environment type is not registered.
        """
        env_type = config.type
        if env_type not in self._registry:
            raise KeyError(
                f"Unknown environment type: {env_type}. Registered types: {list(self._registry)}"
            )
        return self._registry[env_type](config)
