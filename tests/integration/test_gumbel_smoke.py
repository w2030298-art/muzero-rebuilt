"""Smoke test: Gumbel MuZero with TicTacToe."""

from __future__ import annotations

from pathlib import Path

from muzero.cli.train import build_training_components
from muzero.config.loader import ConfigLoader
from muzero.search.gumbel import GumbelPolicy


def test_gumbel_policy_sample_root_actions() -> None:
    """Verify Gumbel root action sampling respects legal mask."""
    import torch

    policy = GumbelPolicy(num_root_samples=4)
    logits = torch.tensor([0.1, 0.5, 0.3, 0.2])
    legal = torch.tensor([True, False, True, True])

    sampled = policy.sample_root_actions(logits, legal, 3)
    assert len(sampled) == 3
    # Action 1 is illegal, should NOT appear
    assert 1 not in sampled


def test_gumbel_tictactoe_components() -> None:
    """Verify Gumbel config builds with TicTacToe."""
    cfg = ConfigLoader().load(Path("configs/tictactoe_gumbel_muzero.yaml"), profile="cpu_debug")
    components = build_training_components(cfg)
    assert components.env.num_players() == 2
    components.env.close()
