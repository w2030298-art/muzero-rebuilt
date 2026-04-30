"""Smoke test: TicTacToe component assembly and MCTS with legal actions."""

from __future__ import annotations

from pathlib import Path

from muzero.cli.train import build_training_components
from muzero.config.loader import ConfigLoader


def test_tictactoe_components_and_search() -> None:
    """Verify TicTacToe components build and MCTS respects legal actions."""
    cfg = ConfigLoader().load(Path("configs/tictactoe_muzero.yaml"), profile="cpu_debug")
    components = build_training_components(cfg)

    assert components.env.num_players() == 2

    # Reset and check legal actions
    ts = components.env.reset(seed=0)
    legal = components.env.legal_actions()
    assert legal is not None
    assert len(legal) == 9
    assert legal.sum() == 9  # All cells available

    # MCTS should return a legal action
    result = components.search.run(
        root_observation=ts.observation,
        legal_actions=legal,
        to_play=ts.to_play,
    )
    assert result.action is not None
    assert 0 <= int(result.action) <= 8  # type: ignore[arg-type]

    components.env.close()
