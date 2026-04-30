"""Smoke test: CartPole MuZero train with cpu_debug profile."""

from __future__ import annotations

from pathlib import Path

from muzero.cli.train import build_training_components
from muzero.config.loader import ConfigLoader
from muzero.execution.self_play_worker import SelfPlayWorker


def test_cartpole_build_and_self_play() -> None:
    """Verify that components build and a self-play episode runs."""
    cfg = ConfigLoader().load(Path("configs/cartpole_muzero.yaml"), profile="cpu_debug")

    components = build_training_components(cfg)

    # Run a self-play episode
    worker = SelfPlayWorker(components.env, components.search, cfg)
    history = worker.run_episode(seed=0)
    assert len(history) > 0

    components.env.close()


def test_cartpole_one_train_step() -> None:
    """Verify one training step runs without error."""
    cfg = ConfigLoader().load(Path("configs/cartpole_muzero.yaml"), profile="cpu_debug")

    components = build_training_components(cfg)

    # Collect one episode for replay
    worker = SelfPlayWorker(components.env, components.search, cfg)
    history = worker.run_episode(seed=0)
    components.replay.add_game(history)

    # Run one train step
    batch = components.replay.sample_batch(batch_size=2, num_unroll_steps=2, td_steps=3)
    result = components.trainer.train_step(batch)
    assert result.step == 1
    assert result.loss_total > 0

    components.env.close()
