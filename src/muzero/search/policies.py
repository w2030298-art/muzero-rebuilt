"""Search policy protocols and implementations for MCTS.

Defines the ``SearchPolicy`` protocol and provides ``PUCTPolicy``
as the default implementation for standard MuZero.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from muzero.config.schema import SearchConfig
from muzero.core.types import Action
from muzero.search.puct import puct_score
from muzero.search.tree_storage import TreeStorage


class SearchPolicy(Protocol):
    """Protocol for search policies used in MCTS child selection.

    Different algorithms (MuZero, Gumbel MuZero) implement this
    to customize the child selection strategy.
    """

    def select_child(self, tree: TreeStorage, node_id: int, config: SearchConfig) -> int:
        """Select the best child of a node according to this policy.

        Args:
            tree: The tree storage containing node data.
            node_id: The parent node to select a child from.
            config: Search configuration for PUCT parameters.

        Returns:
            Index of the selected child node.

        Raises:
            ValueError: If the node has no children.
        """
        ...

    def select_root_action(self, tree: TreeStorage, root_id: int, temperature: float) -> Action:
        """Select an action from the root node's children.

        At ``temperature=0``, returns the most-visited action.
        At ``temperature>0``, samples proportionally to visit counts.

        Args:
            tree: The tree storage containing node data.
            root_id: The root node index.
            temperature: Temperature for action sampling (0 = greedy).

        Returns:
            The selected action.
        """
        ...


class PUCTPolicy:
    """Standard PUCT-based search policy for MuZero.

    Uses the PUCT formula for child selection and visit-count-based
    action selection at the root.
    """

    def select_child(self, tree: TreeStorage, node_id: int, config: SearchConfig) -> int:
        """Select the child with the highest PUCT score.

        Args:
            tree: The tree storage containing node data.
            node_id: The parent node to select a child from.
            config: Search configuration for PUCT parameters.

        Returns:
            Index of the best child node.

        Raises:
            ValueError: If the node has no children (unexpanded).
        """
        children = tree.children(node_id)
        if len(children) == 0:
            raise ValueError("Cannot select from an unexpanded node")

        parent_n = int(tree.visit_count[node_id])

        best_score = float("-inf")
        best_child = children[0]

        for child_id in children:
            child_n = int(tree.visit_count[child_id])
            child_p = float(tree.prior[child_id])

            # Mean Q value for the current player at the child
            to_play = int(tree.to_play[node_id])
            child_q = tree.value(child_id, to_play)

            score = puct_score(
                parent_visit_count=parent_n,
                child_visit_count=child_n,
                child_prior=child_p,
                child_value=child_q,
                pb_c_base=float(config.pb_c_base),
                pb_c_init=float(config.pb_c_init),
            )

            if score > best_score:
                best_score = score
                best_child = child_id

        return int(best_child)

    def select_root_action(self, tree: TreeStorage, root_id: int, temperature: float) -> Action:
        """Select an action from the root node's children.

        Args:
            tree: The tree storage.
            root_id: The root node index.
            temperature: 0 = greedy (most visited), >0 = temperature sampling.

        Returns:
            The selected action.

        Raises:
            ValueError: If the root has no children.
        """
        children = tree.children(root_id)
        if len(children) == 0:
            raise ValueError("Cannot select action from an unexpanded root")

        visits = np.array([tree.visit_count[int(c)] for c in children], dtype=np.float64)

        if temperature == 0.0:
            best_idx = int(np.argmax(visits))
        else:
            # Sample proportionally to visit_count ** (1 / temperature)
            probs = visits ** (1.0 / temperature)
            probs /= probs.sum()
            best_idx = int(np.random.choice(len(children), p=probs))

        action = tree.action_from_parent[int(children[best_idx])]
        if action is None:
            raise ValueError("Root child has no action")
        return action
