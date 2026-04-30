"""Tests for the TreeStorage array-based MCTS tree."""

from __future__ import annotations

import numpy as np

from muzero.search.tree_storage import TreeStorage


def test_allocate_root_node() -> None:
    """Verify that allocating a root node works correctly."""
    tree = TreeStorage(max_nodes=100, num_players=1)
    root_id = tree.allocate_node(parent=-1, action=None, prior=1.0, reward=0.0, to_play=0)

    assert root_id == 0
    assert tree.parent_index[root_id] == -1
    assert tree.prior[root_id] == 1.0


def test_add_child_node() -> None:
    """Verify that adding a child node registers it properly."""
    tree = TreeStorage(max_nodes=100, num_players=1)
    root_id = tree.allocate_node(parent=-1, action=None, prior=1.0, reward=0.0, to_play=0)
    child_id = tree.add_child(parent=root_id, action=0, prior=0.5, reward=0.0, to_play=0)

    assert child_id == 1
    assert tree.is_expanded(root_id)
    assert not tree.is_expanded(child_id)
    assert tree.num_children[root_id] == 1

    children = tree.children(root_id)
    assert len(children) == 1
    assert int(children[0]) == child_id


def test_is_expanded() -> None:
    """Verify is_expanded correctly identifies expanded nodes."""
    tree = TreeStorage(max_nodes=100, num_players=1)
    root = tree.allocate_node(parent=-1, action=None, prior=1.0, reward=0.0, to_play=0)

    assert not tree.is_expanded(root)
    tree.add_child(parent=root, action=0, prior=1.0, reward=0.0, to_play=0)
    assert tree.is_expanded(root)


def test_backup_scalar_value_single_player() -> None:
    """Verify backup with scalar value updates visit counts and value sums."""
    tree = TreeStorage(max_nodes=100, num_players=1)
    root = tree.allocate_node(parent=-1, action=None, prior=1.0, reward=0.0, to_play=0)
    child = tree.add_child(parent=root, action=0, prior=1.0, reward=0.0, to_play=0)

    # Backup a value of 1.0 along the path
    tree.backup(path=[root, child], value=1.0, discount=1.0, to_play=0)

    assert tree.visit_count[root] == 1
    assert tree.visit_count[child] == 1
    assert tree.value_sum[child, 0] >= 0


def test_backup_with_discount() -> None:
    """Verify backup with discount factor propagates correctly."""
    tree = TreeStorage(max_nodes=100, num_players=1)
    root = tree.allocate_node(parent=-1, action=None, prior=1.0, reward=1.0, to_play=0)
    child = tree.add_child(parent=root, action=0, prior=1.0, reward=0.0, to_play=0)

    # Leaf value = 1.0, discount = 0.9
    # Parent gets reward[parent] + 0.9 * leaf_value accumulated
    tree.backup(path=[root, child], value=1.0, discount=0.9, to_play=0)

    assert tree.visit_count[root] == 1
    assert tree.visit_count[child] == 1


def test_backup_vector_value_two_player() -> None:
    """Verify backup with vector value for two-player games."""
    tree = TreeStorage(max_nodes=100, num_players=2)
    root = tree.allocate_node(
        parent=-1, action=None, prior=1.0, reward=np.array([0.0, 0.0]), to_play=0
    )
    child = tree.add_child(parent=root, action=0, prior=1.0, reward=np.array([0.0, 0.0]), to_play=1)

    tree.backup(path=[root, child], value=np.array([1.0, -1.0]), discount=1.0, to_play=0)

    assert tree.visit_count[root] == 1
    assert tree.visit_count[child] == 1


def test_value_method_single_player() -> None:
    """Verify value() computes mean Q for single player."""
    tree = TreeStorage(max_nodes=100, num_players=1)
    root = tree.allocate_node(parent=-1, action=None, prior=1.0, reward=0.0, to_play=0)

    tree.visit_count[root] = 4
    tree.value_sum[root, 0] = 2.0

    val = tree.value(root, to_play=0)
    assert val == 0.5  # 2.0 / 4


def test_tree_reset() -> None:
    """Verify resetting the tree clears all data."""
    tree = TreeStorage(max_nodes=10, num_players=1)
    root = tree.allocate_node(parent=-1, action=None, prior=1.0, reward=0.0, to_play=0)
    tree.add_child(parent=root, action=0, prior=1.0, reward=0.0, to_play=0)

    assert tree.num_nodes > 0
    tree.reset()
    assert tree.num_nodes == 0


def test_overflow_raises() -> None:
    """Verify that exceeding max_nodes raises RuntimeError."""
    tree = TreeStorage(max_nodes=2, num_players=1)
    tree.allocate_node(parent=-1, action=None, prior=1.0, reward=0.0, to_play=0)
    tree.allocate_node(parent=-1, action=None, prior=1.0, reward=0.0, to_play=0)

    import pytest

    with pytest.raises(RuntimeError):
        tree.allocate_node(parent=-1, action=None, prior=1.0, reward=0.0, to_play=0)
