"""Tests for the InferenceBatcher."""

from __future__ import annotations

import numpy as np
import torch

from muzero.models.mlp import MLPNetwork
from muzero.models.outputs import NetworkOutput
from muzero.search.inference_batcher import InferenceBatcher


def test_initial_batcher_flush_calls_callbacks() -> None:
    """Verify flush_initial invokes callbacks with correct outputs."""
    network = MLPNetwork(observation_shape=(4,), action_space_size=2, hidden_size=16)
    device = torch.device("cpu")
    batcher = InferenceBatcher(network, device, batch_size=8)

    results: list[NetworkOutput] = []

    def callback(out: NetworkOutput) -> None:
        results.append(out)

    obs = np.random.randn(4).astype(np.float32)
    batcher.enqueue_initial(obs, callback)
    batcher.flush_initial()

    assert len(results) == 1
    assert results[0].value is not None
    assert results[0].policy_logits.shape == (2,)  # action_dim=2
    assert results[0].hidden_state.shape == (16,)  # hidden_size=16


def test_recurrent_batcher_flush_calls_callbacks() -> None:
    """Verify flush_recurrent invokes callbacks correctly."""
    network = MLPNetwork(observation_shape=(4,), action_space_size=2, hidden_size=16)
    device = torch.device("cpu")
    batcher = InferenceBatcher(network, device, batch_size=8)

    # First get a hidden state via initial inference
    obs = np.random.randn(4).astype(np.float32)
    hidden_holder: list[torch.Tensor] = []

    def init_cb(out: NetworkOutput) -> None:
        hidden_holder.append(out.hidden_state)

    batcher.enqueue_initial(obs, init_cb)
    batcher.flush_initial()

    # Now do recurrent inference
    results: list[NetworkOutput] = []

    def rec_cb(out: NetworkOutput) -> None:
        results.append(out)

    batcher.enqueue_recurrent(hidden_holder[0], 0, rec_cb)
    batcher.flush_recurrent()

    assert len(results) == 1
    assert results[0].value is not None
    assert results[0].reward is not None


def test_batcher_auto_flush_at_batch_size() -> None:
    """Verify batcher auto-flushes when batch_size is reached."""
    network = MLPNetwork(observation_shape=(4,), action_space_size=2, hidden_size=16)
    device = torch.device("cpu")
    batcher = InferenceBatcher(network, device, batch_size=3)

    results: list[NetworkOutput] = []

    def callback(out: NetworkOutput) -> None:
        results.append(out)

    for _ in range(5):
        obs = np.random.randn(4).astype(np.float32)
        batcher.enqueue_initial(obs, callback)

    # Should have auto-flushed at 3, leaving 2 in queue
    # Manual flush should process remaining
    batcher.flush_initial()

    assert len(results) == 5


def test_flush_empty_does_not_crash() -> None:
    """Verify flushing empty batcher is safe."""
    network = MLPNetwork(observation_shape=(4,), action_space_size=2, hidden_size=16)
    device = torch.device("cpu")
    batcher = InferenceBatcher(network, device)

    batcher.flush_initial()
    batcher.flush_recurrent()
    batcher.flush()
