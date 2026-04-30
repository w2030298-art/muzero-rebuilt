"""Array-based tree storage for MCTS.

Uses NumPy arrays for all tree data to enable efficient CPU operations
and batch inference integration. Nodes are stored contiguously in flat arrays,
with children tracked via linked-list-style indices.
"""

from __future__ import annotations

import numpy as np

from muzero.core.types import Action, Value


class TreeStorage:
    """Array-based storage for MCTS tree nodes.

    All node data is stored in pre-allocated NumPy arrays for performance.
    Child relationships use a first-child index + count pattern.

    Args:
        max_nodes: Maximum number of nodes to allocate.
        num_players: Number of players (1 for single, 2+ for multi).
    """

    def __init__(
        self, max_nodes: int, num_players: int, action_shape: tuple[int, ...] = ()
    ) -> None:
        self._max_nodes = max_nodes
        self._num_players = num_players
        self._n_val = 1 if num_players == 1 else num_players
        self._next_node_id = 0

        # Core arrays
        self.visit_count: np.ndarray = np.zeros(max_nodes, dtype=np.int32)
        self.value_sum: np.ndarray = np.zeros((max_nodes, self._n_val), dtype=np.float32)
        self.prior: np.ndarray = np.zeros(max_nodes, dtype=np.float32)
        self.reward: np.ndarray = np.zeros((max_nodes, self._n_val), dtype=np.float32)
        self.parent_index: np.ndarray = np.full(max_nodes, -1, dtype=np.int32)
        self.first_child_index: np.ndarray = np.full(max_nodes, -1, dtype=np.int32)
        self.num_children: np.ndarray = np.zeros(max_nodes, dtype=np.int32)
        self.to_play: np.ndarray = np.full(max_nodes, -1, dtype=np.int32)
        self.hidden_state_id: np.ndarray = np.full(max_nodes, -1, dtype=np.int32)

        # Child tracking (Python lists for variable-length children)
        self.child_indices: list[list[int]] = [[] for _ in range(max_nodes)]  # type: ignore[misc]
        self.action_from_parent: list[Action | None] = [None] * max_nodes

    def allocate_node(
        self,
        parent: int,
        action: Action | None,
        prior: float,
        reward: Value,
        to_play: int,
    ) -> int:
        """Allocate a new node in the tree.

        Args:
            parent: Parent node index, or -1 for root.
            action: Action from parent to this node.
            prior: Prior probability from the policy network.
            reward: Reward value (scalar or vector).
            to_play: Player index at this node.

        Returns:
            The new node's index.

        Raises:
            RuntimeError: If the tree has reached its max capacity.
        """
        if self._next_node_id >= self._max_nodes:
            raise RuntimeError(f"TreeStorage overflow: max_nodes={self._max_nodes} reached.")

        node_id = self._next_node_id
        self._next_node_id += 1

        self.parent_index[node_id] = parent
        self.action_from_parent[node_id] = action
        self.prior[node_id] = prior
        self.to_play[node_id] = to_play

        # Set reward
        if isinstance(reward, np.ndarray):
            self.reward[node_id] = reward
        else:
            self.reward[node_id, 0] = float(reward)

        return node_id

    def add_child(
        self,
        parent: int,
        action: Action,
        prior: float,
        reward: Value,
        to_play: int,
    ) -> int:
        """Allocate a child node and register it with the parent.

        Args:
            parent: Parent node index.
            action: Action from parent to child.
            prior: Prior probability.
            reward: Reward value.
            to_play: Player index at the child node.

        Returns:
            The child node's index.
        """
        child_id = self.allocate_node(parent, action, prior, reward, to_play)
        self.child_indices[parent].append(child_id)
        self.num_children[parent] = len(self.child_indices[parent])
        if self.first_child_index[parent] == -1:
            self.first_child_index[parent] = child_id
        return child_id

    def children(self, node_id: int) -> np.ndarray:
        """Return the child node indices of a node.

        Args:
            node_id: Node index.

        Returns:
            Array of child node indices.
        """
        return np.array(self.child_indices[node_id], dtype=np.int32)

    def is_expanded(self, node_id: int) -> bool:
        """Check if a node has been expanded (has children).

        Args:
            node_id: Node index.

        Returns:
            True if the node has children.
        """
        return self.num_children[node_id] > 0

    def value(self, node_id: int, to_play: int) -> float:
        """Return the expected value for a specific player at this node.

        For single-player: returns the scalar value.
        For multi-player: returns value_sum for the specified player,
        divided by visit count.

        Args:
            node_id: Node index.
            to_play: Player index to get value for.

        Returns:
            Mean value for the specified player.
        """
        vc = self.visit_count[node_id]
        if vc == 0:
            return 0.0
        if self._num_players == 1:
            return float(self.value_sum[node_id, 0]) / vc
        return float(self.value_sum[node_id, to_play]) / vc

    def backup(
        self,
        path: list[int],
        value: Value,
        discount: float,
        to_play: int,
    ) -> None:
        """Backup values along a search path using discounted accumulation.

        For each node in the path, the value is accumulated into ``value_sum``
        and the visit count is incremented.

        Args:
            path: List of node indices from root to leaf.
            value: Value to backup (scalar or vector).
            discount: Discount factor for value propagation.
            to_play: Player index (unused - reserved for perspective projection).
        """
        if isinstance(value, np.ndarray):
            val_arr = value.astype(np.float32)
        else:
            val_arr = np.array([float(value)], dtype=np.float32)
            if self._n_val > 1:
                val_arr = np.full(self._n_val, float(value), dtype=np.float32)

        for node_id in reversed(path):
            self.visit_count[node_id] += 1

            # Accumulate reward + discounted value
            reward = self.reward[node_id]
            if self._n_val == 1:
                self.value_sum[node_id, 0] += float(reward[0]) + discount * float(val_arr[0])  # type: ignore[call-overload]
            else:
                self.value_sum[node_id] += reward + discount * val_arr

            # Propagate: value becomes reward for parent
            val_arr = reward + discount * val_arr

    @property
    def num_nodes(self) -> int:
        """Total number of allocated nodes."""
        return self._next_node_id

    def reset(self) -> None:
        """Reset the tree to empty state (reuses allocated arrays)."""
        self._next_node_id = 0
        self.visit_count.fill(0)
        self.value_sum.fill(0)
        self.prior.fill(0)
        self.reward.fill(0)
        self.parent_index.fill(-1)
        self.first_child_index.fill(-1)
        self.num_children.fill(0)
        self.to_play.fill(-1)
        self.hidden_state_id.fill(-1)
        for i in range(self._max_nodes):
            self.child_indices[i].clear()
            self.action_from_parent[i] = None
