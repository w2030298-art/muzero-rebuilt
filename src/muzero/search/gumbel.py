"""Gumbel MuZero search policy — root action sampling with Gumbel noise."""

from __future__ import annotations

import numpy as np
import torch

from muzero.config.schema import SearchConfig
from muzero.core.types import Action
from muzero.search.policies import PUCTPolicy
from muzero.search.tree_storage import TreeStorage


class GumbelPolicy(PUCTPolicy):
    """Search policy using Gumbel noise for root action sampling.

    Root actions are sampled using logits + Gumbel(0,1) top-k.
    Non-root child selection uses PUCT (delegates to parent class).

    Args:
        num_root_samples: Number of root actions to sample via Gumbel.
    """

    def __init__(self, num_root_samples: int = 16) -> None:
        super().__init__()
        self._num_root_samples = num_root_samples

    def sample_root_actions(
        self,
        policy_logits: torch.Tensor,
        legal_mask: torch.Tensor | None,
        num_samples: int,
    ) -> np.ndarray:
        """Sample root actions using Gumbel noise top-k.

        Args:
            policy_logits: Raw logits, shape [action_dim].
            legal_mask: Legal action mask or None.
            num_samples: Number of actions to sample.

        Returns:
            Array of sampled action indices.
        """
        logits = policy_logits.clone()
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float("-inf"))

        gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-8)))
        noisy = logits + gumbel

        k = min(num_samples, logits.shape[0])
        _, indices = torch.topk(noisy, k)
        return indices.cpu().numpy()

    def select_child(self, tree: TreeStorage, node_id: int, config: SearchConfig) -> int:
        """Select child using PUCT (delegated)."""
        return super().select_child(tree, node_id, config)

    def sequential_halving(self, tree: TreeStorage, root_id: int) -> Action:
        """Select root action: most-visited among candidates (simplified)."""
        children = tree.children(root_id)
        if len(children) == 0:
            raise ValueError("Root has no children")

        best = children[0]
        best_v = tree.visit_count[int(best)]
        for c in children:
            vc = tree.visit_count[int(c)]
            if vc > best_v:
                best_v = vc
                best = c

        action = tree.action_from_parent[int(best)]
        if action is None:
            raise ValueError("No action for best child")
        return action
