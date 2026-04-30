"""Tests for the ReplayBuffer, PrioritizedReplayBuffer, and ReanalyzeQueue."""

from __future__ import annotations

import numpy as np

from muzero.config.schema import ReplayConfig
from muzero.core.game_history import GameHistory
from muzero.core.types import SearchResult, TimeStep
from muzero.replay.buffer import ReplayBuffer
from muzero.replay.prioritized import PrioritizedReplayBuffer
from muzero.replay.reanalyze import ReanalyzeQueue, ReplayItemRef
from muzero.replay.target_builder import TargetBuilder


def _make_game_history(length: int = 3) -> GameHistory:
    """Create a simple game history for testing."""
    history = GameHistory()
    obs = np.zeros(4, dtype=np.float32)

    ts0 = TimeStep(observation=obs, reward=0.0, terminated=False, truncated=False, to_play=0)
    history.append_initial(ts0)

    for i in range(length):
        ts = TimeStep(
            observation=obs + float(i + 1),
            reward=float(i + 1),
            terminated=(i == length - 1),
            truncated=False,
            to_play=0,
        )
        result = SearchResult(
            action=0,
            root_value=float(i + 1),
            visit_counts=np.array([0.5, 0.5], dtype=np.float32),
            policy_target=np.array([0.5, 0.5], dtype=np.float32),
        )
        history.append(action=0, timestep=ts, search_result=result)

    return history


def test_replay_buffer_add_and_len() -> None:
    """Verify adding games increases buffer length."""
    config = ReplayConfig(capacity=10)
    tb = TargetBuilder(discount=0.997)
    buf = ReplayBuffer(config, tb)

    assert len(buf) == 0
    buf.add_game(_make_game_history(3))
    assert len(buf) == 1


def test_replay_buffer_sample_batch() -> None:
    """Verify sample_batch returns a TrainingBatch with correct shapes."""
    config = ReplayConfig(capacity=10)
    tb = TargetBuilder(discount=0.997)
    buf = ReplayBuffer(config, tb)

    # Add a few games
    for _ in range(5):
        buf.add_game(_make_game_history(3))

    batch = buf.sample_batch(batch_size=4, num_unroll_steps=2, td_steps=1)
    assert batch.observations is not None


def test_replay_buffer_ring_overflow() -> None:
    """Verify ring buffer behavior when capacity is exceeded."""
    config = ReplayConfig(capacity=3)
    tb = TargetBuilder(discount=0.997)
    buf = ReplayBuffer(config, tb)

    for _ in range(5):
        buf.add_game(_make_game_history(2))

    assert len(buf) == 3
    assert buf.sample_batch(batch_size=2, num_unroll_steps=1, td_steps=1) is not None


def test_prioritized_replay_buffer() -> None:
    """Verify prioritized buffer samples and returns importance weights."""
    config = ReplayConfig(capacity=10, prioritized=True, alpha=0.6, beta=0.4)
    tb = TargetBuilder(discount=0.997)
    buf = PrioritizedReplayBuffer(config, tb)

    for _ in range(5):
        buf.add_game(_make_game_history(3))

    batch = buf.sample_batch(batch_size=4, num_unroll_steps=2, td_steps=1)
    assert batch.importance_weights is not None
    assert batch.indices is not None
    assert len(batch.importance_weights) == 4  # type: ignore[arg-type]


def test_prioritized_update_priorities() -> None:
    """Verify priority updates don't crash."""
    config = ReplayConfig(capacity=10, prioritized=True)
    tb = TargetBuilder(discount=0.997)
    buf = PrioritizedReplayBuffer(config, tb)

    for _ in range(5):
        buf.add_game(_make_game_history(3))

    buf.update_priorities(np.array([0, 1]), np.array([0.5, 0.8]))


def test_reanalyze_queue() -> None:
    """Verify reanalyze queue enqueue and dequeue."""
    q = ReanalyzeQueue()
    refs = [ReplayItemRef(0, 0), ReplayItemRef(1, 0)]
    q.enqueue(refs)
    assert len(q) == 2

    items = q.dequeue(1)
    assert len(items) == 1
    assert len(q) == 1
    assert items[0].game_index == 0


def test_reanalyze_queue_empty_dequeue() -> None:
    """Verify dequeue from empty queue returns empty list."""
    q = ReanalyzeQueue()
    assert q.dequeue(5) == []
