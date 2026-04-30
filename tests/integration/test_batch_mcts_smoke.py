"""Smoke test: BatchMCTS with shared InferenceBatcher."""

from __future__ import annotations

import numpy as np
import torch

from muzero.config.schema import SearchConfig
from muzero.models.mlp import MLPNetwork
from muzero.search.action_sampler import DiscreteActionSampler
from muzero.search.batch_mcts import BatchMCTS
from muzero.search.inference_batcher import InferenceBatcher
from muzero.search.mcts import MCTS, SearchRequest
from muzero.search.policies import PUCTPolicy


def test_batch_mcts_runs() -> None:
    """Verify BatchMCTS can run multiple search requests."""
    network = MLPNetwork(observation_shape=(4,), action_space_size=2, hidden_size=16)
    device = torch.device("cpu")
    config = SearchConfig(num_simulations=4, discount=1.0)

    batcher = InferenceBatcher(network, device, batch_size=8)
    sampler = DiscreteActionSampler()
    policy = PUCTPolicy()

    def mcts_factory() -> MCTS:
        return MCTS(
            network=network,
            config=config,
            action_sampler=sampler,
            search_policy=policy,
            device=device,
            num_players=1,
            action_dim=2,
            inference_batcher=batcher,
        )

    batch_mcts = BatchMCTS(mcts_factory, batcher)

    requests = [
        SearchRequest(
            observation=np.random.randn(4).astype(np.float32),
            legal_actions=np.ones(2, dtype=bool),
            to_play=0,
        )
        for _ in range(4)
    ]

    results = batch_mcts.run_batch(requests)
    assert len(results) == 4

    for r in results:
        assert r.action in [0, 1]
        assert r.visit_counts.shape == (2,)
        assert abs(float(r.policy_target.sum()) - 1.0) < 1e-5

    batch_mcts.flush_inference()
