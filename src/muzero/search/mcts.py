"""Standard MuZero MCTS (Monte Carlo Tree Search).

Implements the full MCTS algorithm with root expansion, selection,
leaf expansion, backup, Dirichlet noise, and policy target construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from muzero.config.schema import SearchConfig
from muzero.core.types import Action, SearchResult, Value
from muzero.models.outputs import NetworkOutput
from muzero.models.protocol import MuZeroNetworkProtocol
from muzero.search.action_sampler import ActionSampler
from muzero.search.inference_batcher import InferenceBatcher
from muzero.search.policies import SearchPolicy
from muzero.search.tree_storage import TreeStorage


@dataclass(slots=True)
class SearchRequest:
    """A single MCTS search request.

    Attributes:
        observation: Root observation, as a numpy array.
        legal_actions: Legal action mask, or None.
        to_play: Current player index (0-based).
    """

    observation: np.ndarray
    legal_actions: np.ndarray | None
    to_play: int


class MCTS:
    """Standard MuZero Monte Carlo Tree Search.

    Performs ``num_simulations`` cycles of select → expand → backup
    from a single root observation.

    Args:
        network: MuZero network for initial and recurrent inference.
        config: Search configuration (simulations, PUCT params, etc.).
        action_sampler: Sampler for choosing actions to expand.
        search_policy: Policy for child selection and action selection.
        device: Torch device for tensor operations.
    """

    def __init__(
        self,
        network: MuZeroNetworkProtocol,
        config: SearchConfig,
        action_sampler: ActionSampler,
        search_policy: SearchPolicy,
        device: torch.device,
        num_players: int = 1,
        action_dim: int | None = None,
        inference_batcher: InferenceBatcher | None = None,
    ) -> None:
        self._network = network
        self._config = config
        self._action_sampler = action_sampler
        self._search_policy = search_policy
        self._device = device
        self._num_players = num_players
        self._action_dim = action_dim
        self._batcher = inference_batcher
        self._hidden_states: list[torch.Tensor] = []

    def _initial_inference(self, obs_tensor: torch.Tensor) -> NetworkOutput:
        """Run initial inference, optionally through the batcher."""
        if self._batcher is not None:
            result_holder: list[NetworkOutput] = []

            def callback(out: NetworkOutput) -> None:
                result_holder.append(out)

            self._batcher.enqueue_initial(obs_tensor.squeeze(0).cpu().numpy(), callback)
            self._batcher.flush_initial()
            return result_holder[0]

        with torch.no_grad():
            return self._network.initial_inference(obs_tensor)

    def _recurrent_inference(
        self, hidden_state: torch.Tensor, action_tensor: torch.Tensor
    ) -> NetworkOutput:
        """Run recurrent inference, optionally through the batcher."""
        if self._batcher is not None:
            result_holder: list[NetworkOutput] = []

            def callback(out: NetworkOutput) -> None:
                result_holder.append(out)

            # hidden_state is [1, H], squeeze to [H] for batcher
            hs_cpu = hidden_state.squeeze(0).cpu()
            self._batcher.enqueue_recurrent(
                hs_cpu,
                int(action_tensor.item()),
                callback,
            )
            self._batcher.flush_recurrent()
            return result_holder[0]

        with torch.no_grad():
            return self._network.recurrent_inference(hidden_state, action_tensor)

    def run(
        self,
        root_observation: np.ndarray,
        legal_actions: np.ndarray | None,
        to_play: int,
    ) -> SearchResult:
        """Run MCTS from a root observation.

        Args:
            root_observation: Observation as a numpy array.
            legal_actions: Legal action mask, or None if all actions are legal.
            to_play: Current player index.

        Returns:
            SearchResult with selected action, visit counts, policy target.
        """
        num_actions: int = self._get_action_dim(
            legal_actions, fallback=self._action_dim if self._action_dim else 10
        )
        max_nodes: int = self._config.num_simulations * (num_actions + 1) + 100

        num_players: int = self._num_players
        tree = TreeStorage(max_nodes=max_nodes, num_players=num_players)

        # Root expansion
        root_id = self._expand_root(tree, root_observation, legal_actions, to_play)

        # Add Dirichlet noise to root priors
        self._add_root_exploration_noise(tree, root_id)

        # Simulation loop
        for _ in range(self._config.num_simulations):
            self._run_simulation(tree, root_id)

        # Build policy target
        policy_target = self._build_policy_target(tree, root_id, num_actions)

        # Select action
        action = self._search_policy.select_root_action(tree, root_id, self._config.temperature)

        visit_counts = np.zeros(num_actions, dtype=np.float32)
        children = tree.children(root_id)
        for child_id in children:
            cid = int(child_id)
            child_action = tree.action_from_parent[cid]
            if child_action is not None and isinstance(child_action, (int, np.integer)):
                visit_counts[int(child_action)] = float(tree.visit_count[cid])

        root_value: Value
        if num_players == 1:
            root_value = float(tree.value(root_id, 0))
        else:
            root_value = np.array(
                [tree.value(root_id, p) for p in range(num_players)],
                dtype=np.float32,
            )

        return SearchResult(
            action=action,
            root_value=root_value,
            visit_counts=visit_counts,
            policy_target=policy_target,
            search_depth=self._config.num_simulations,
            num_expanded_nodes=tree.num_nodes,
        )

    def _expand_root(
        self,
        tree: TreeStorage,
        root_observation: np.ndarray,
        legal_actions: np.ndarray | None,
        to_play: int,
    ) -> int:
        """Create root node and expand its children.

        Args:
            tree: Tree storage.
            root_observation: Root observation.
            legal_actions: Legal action mask.
            to_play: Current player.

        Returns:
            Root node index.
        """
        # Convert observation to batched tensor
        obs_tensor = torch.from_numpy(root_observation).float().unsqueeze(0).to(self._device)
        output = self._initial_inference(obs_tensor)

        # Store hidden state
        hidden_id = len(self._hidden_states)
        out_hidden = output.hidden_state
        if out_hidden.dim() == 1:
            out_hidden = out_hidden.unsqueeze(0)
        self._hidden_states.append(out_hidden[0].cpu())

        # Create root node
        root_id = tree.allocate_node(parent=-1, action=None, prior=1.0, reward=0.0, to_play=to_play)
        tree.hidden_state_id[root_id] = hidden_id

        # Sample actions for expansion
        policy_output = output.policy_logits
        if policy_output.dim() == 1:
            policy_output = policy_output.unsqueeze(0)
        hidden = output.hidden_state
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)

        legal_mask_tensor: torch.Tensor | None = None
        if legal_actions is not None:
            legal_mask_tensor = torch.from_numpy(legal_actions).bool()

        sample_batch = self._action_sampler.sample(
            policy_output=policy_output,
            hidden_state=hidden,
            num_samples=policy_output.shape[1],
            legal_action_mask=(
                legal_mask_tensor.unsqueeze(0) if legal_mask_tensor is not None else None
            ),
        )

        # Create child nodes
        child_to_play = (to_play + 1) % self._num_players
        for k in range(sample_batch.actions.shape[1]):
            action = sample_batch.actions[0, k]
            prior = sample_batch.priors[0, k]

            action_val = self._extract_action(action)

            tree.add_child(
                parent=root_id,
                action=action_val,
                prior=float(prior.item()),
                reward=0.0,
                to_play=child_to_play,
            )

        return root_id

    def _run_simulation(self, tree: TreeStorage, root_id: int) -> None:
        """Run one simulation: select → expand → backup.

        Args:
            tree: Tree storage.
            root_id: Root node index.
        """
        path = self._select_path(tree, root_id)
        leaf_id = path[-1]
        leaf_value = self._expand_leaf(tree, leaf_id)
        self._backup(tree, path, leaf_value)

    def _select_path(self, tree: TreeStorage, root_id: int) -> list[int]:
        """Select a path from root to an unexpanded leaf.

        Args:
            tree: Tree storage.
            root_id: Root node index.

        Returns:
            List of node indices from root to leaf (inclusive).
        """
        path: list[int] = [root_id]
        node_id = root_id

        while tree.is_expanded(node_id) and tree.num_children[node_id] > 0:
            node_id = self._search_policy.select_child(tree, node_id, self._config)
            path.append(node_id)

        return path

    def _expand_leaf(self, tree: TreeStorage, leaf_id: int) -> np.ndarray | float:
        """Expand a leaf node using the network and return its value.

        Args:
            tree: Tree storage.
            leaf_id: Leaf node index.

        Returns:
            The value of the leaf node (scalar or vector).
        """
        parent_id: int = int(tree.parent_index[leaf_id])

        if parent_id < 0:
            # Root: value is already set from initial_inference
            return 0.0

        # Get parent hidden state
        parent_hidden_id: int = int(tree.hidden_state_id[parent_id])
        hidden_state = self._hidden_states[parent_hidden_id].unsqueeze(0).to(self._device)

        # Get action
        action = tree.action_from_parent[leaf_id]
        if action is None:
            return 0.0
        action_tensor = torch.tensor([int(action)], device=self._device)
        output = self._recurrent_inference(hidden_state, action_tensor)

        # Store hidden state
        hidden_id = len(self._hidden_states)
        out_hidden = output.hidden_state
        if out_hidden.dim() == 1:
            out_hidden = out_hidden.unsqueeze(0)
        self._hidden_states.append(out_hidden[0].cpu())
        tree.hidden_state_id[leaf_id] = hidden_id

        # Get value
        num_players = self._num_players
        out_value = output.value
        if out_value.dim() == 0:
            leaf_value: np.ndarray | float = float(out_value.item())
        elif num_players == 1:
            leaf_value = (
                float(out_value[0].item()) if out_value.dim() > 1 else float(out_value.item())
            )
        else:
            leaf_value = (
                out_value[0].cpu().numpy() if out_value.dim() > 1 else out_value.cpu().numpy()
            )

        # Expand children of leaf
        policy_output = output.policy_logits
        if policy_output.dim() == 1:
            policy_output = policy_output.unsqueeze(0)
        hidden = output.hidden_state
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        num_actions = policy_output.shape[1]
        leaf_to_play = int(tree.to_play[leaf_id])

        sample_batch = self._action_sampler.sample(
            policy_output=policy_output,
            hidden_state=hidden,
            num_samples=num_actions,
            legal_action_mask=None,
        )

        child_to_play = (leaf_to_play + 1) % num_players
        child_reward: Value
        out_reward = output.reward
        if out_reward.dim() == 0:
            child_reward = float(out_reward.item())
        elif num_players == 1:
            child_reward = (
                float(out_reward[0].item()) if out_reward.dim() > 1 else float(out_reward.item())
            )
        else:
            child_reward = (
                out_reward[0].cpu().numpy() if out_reward.dim() > 1 else out_reward.cpu().numpy()
            )

        for k in range(sample_batch.actions.shape[1]):
            child_action = sample_batch.actions[0, k]
            child_prior = sample_batch.priors[0, k]

            child_action_val: Action = self._extract_action(child_action)

            tree.add_child(
                parent=leaf_id,
                action=child_action_val,
                prior=float(child_prior.item()),
                reward=child_reward,
                to_play=child_to_play,
            )

        return leaf_value

    def _backup(
        self,
        tree: TreeStorage,
        path: list[int],
        value: np.ndarray | float,
    ) -> None:
        """Backup value along the search path.

        Args:
            tree: Tree storage.
            path: Node indices from root to leaf.
            value: Value to backup.
        """
        tree.backup(
            path=path,
            value=value,
            discount=self._config.discount,
            to_play=0,
        )

    def _add_root_exploration_noise(self, tree: TreeStorage, root_id: int) -> None:
        """Add Dirichlet noise to root children priors.

        Mixes original priors with Dirichlet noise according to
        ``root_exploration_fraction``.

        Args:
            tree: Tree storage.
            root_id: Root node index.
        """
        children = tree.children(root_id)
        n_children = len(children)
        if n_children == 0:
            return

        alpha = self._config.dirichlet_alpha
        frac = self._config.root_exploration_fraction

        noise = np.random.dirichlet([alpha] * n_children).astype(np.float32)

        for i, child_id in enumerate(children):
            tree.prior[int(child_id)] = (1.0 - frac) * tree.prior[int(child_id)] + frac * noise[i]

    def _build_policy_target(self, tree: TreeStorage, root_id: int, num_actions: int) -> np.ndarray:
        """Build policy target from root child visit counts.

        Creates a probability distribution over all actions. Actions that
        were not expanded receive zero probability.

        Args:
            tree: Tree storage.
            root_id: Root node index.
            num_actions: Total number of possible actions.

        Returns:
            Policy target array of shape ``[num_actions]`` summing to 1.
        """
        target = np.zeros(num_actions, dtype=np.float32)
        children = tree.children(root_id)

        total_visits = 0
        action_visits: dict[int, int] = {}
        for child_id in children:
            cid = int(child_id)
            action = tree.action_from_parent[cid]
            if action is not None and isinstance(action, (int, np.integer)):
                visits = int(tree.visit_count[cid])
                action_visits[int(action)] = visits
                total_visits += visits

        if total_visits > 0:
            for action_idx, visits in action_visits.items():
                target[action_idx] = float(visits) / total_visits
        else:
            # Uniform if no visits
            target[:] = 1.0 / num_actions

        return target

    @staticmethod
    def _extract_action(action_tensor: torch.Tensor) -> Action:
        """Extract action from a tensor (handles discrete and continuous).

        For discrete (0-D tensor): returns int.
        For continuous (1-D tensor): returns numpy array.
        """
        if action_tensor.numel() == 1:
            return int(action_tensor.item())
        return action_tensor.cpu().numpy()

    @staticmethod
    def _get_action_dim(legal_actions: np.ndarray | None, fallback: int = 10) -> int:
        """Get the action dimension from a legal action mask.

        Args:
            legal_actions: Legal action mask or None.
            fallback: Default action dimension if mask is None.

        Returns:
            Number of possible actions.
        """
        if legal_actions is not None:
            return int(legal_actions.shape[0])
        return fallback
