"""Inference batcher for batch GPU inference in MCTS.

Collects individual inference requests from MCTS tree search and
batch-processes them to maximize GPU utilization.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from muzero.core.types import Action
from muzero.models.outputs import NetworkOutput
from muzero.models.protocol import MuZeroNetworkProtocol


@dataclass(slots=True)
class _InitialRequest:
    observation: np.ndarray
    callback: Callable[[NetworkOutput], None]


@dataclass(slots=True)
class _RecurrentRequest:
    hidden_state: torch.Tensor
    action: Action
    callback: Callable[[NetworkOutput], None]


class InferenceBatcher:
    """Batched inference queue for MCTS.

    Collects individual initial/recurrent inference requests and
    processes them in batches for efficient GPU utilization.

    Args:
        network: MuZero network for inference.
        device: Torch device for tensor operations.
        batch_size: Maximum batch size for auto-flush.
        precision: AMP precision mode (fp32, amp_fp16, amp_bf16).
    """

    def __init__(
        self,
        network: MuZeroNetworkProtocol,
        device: torch.device,
        batch_size: int = 16,
        precision: str = "fp32",
    ) -> None:
        self._network = network
        self._device = device
        self._batch_size = batch_size
        self._precision = precision

        self._initial_queue: list[_InitialRequest] = []
        self._recurrent_queue: list[_RecurrentRequest] = []

        use_amp = precision != "fp32" and device.type == "cuda"
        self._amp_dtype = torch.float16
        if precision == "amp_bf16":
            self._amp_dtype = torch.bfloat16
        self._use_amp = use_amp

    def enqueue_initial(
        self,
        observation: np.ndarray,
        callback: Callable[[NetworkOutput], None],
    ) -> None:
        """Enqueue an initial inference request.

        Auto-flushes if the queue reaches ``batch_size``.

        Args:
            observation: Single observation array (no batch dim).
            callback: Called with the single-sample NetworkOutput.
        """
        self._initial_queue.append(_InitialRequest(observation, callback))
        if len(self._initial_queue) >= self._batch_size:
            self.flush_initial()

    def enqueue_recurrent(
        self,
        hidden_state: torch.Tensor,
        action: Action,
        callback: Callable[[NetworkOutput], None],
    ) -> None:
        """Enqueue a recurrent inference request.

        Auto-flushes if the queue reaches ``batch_size``.

        Args:
            hidden_state: Single hidden state tensor (no batch dim).
            action: Action taken.
            callback: Called with the single-sample NetworkOutput.
        """
        self._recurrent_queue.append(_RecurrentRequest(hidden_state, action, callback))
        if len(self._recurrent_queue) >= self._batch_size:
            self.flush_recurrent()

    def flush_initial(self) -> None:
        """Process all queued initial inference requests as a batch."""
        if not self._initial_queue:
            return

        obs_list = [req.observation for req in self._initial_queue]
        obs_batch = np.stack(obs_list)
        obs_tensor = torch.from_numpy(obs_batch).float().to(self._device)

        outputs = self._run_inference(lambda: self._network.initial_inference(obs_tensor))

        for i, req in enumerate(self._initial_queue):
            # Extract single-sample output
            single = NetworkOutput(
                value=outputs.value[i],
                reward=outputs.reward[i],
                policy_logits=outputs.policy_logits[i],
                hidden_state=outputs.hidden_state[i],
                value_prefix=(
                    outputs.value_prefix[i] if outputs.value_prefix is not None else None
                ),
                projection=(outputs.projection[i] if outputs.projection is not None else None),
            )
            req.callback(single)

        self._initial_queue.clear()

    def flush_recurrent(self) -> None:
        """Process all queued recurrent inference requests as a batch."""
        if not self._recurrent_queue:
            return

        hidden_list = [req.hidden_state.unsqueeze(0) for req in self._recurrent_queue]
        hidden_batch = torch.cat(hidden_list, dim=0).to(self._device)

        action_list = [
            torch.tensor([int(req.action)], device=self._device) for req in self._recurrent_queue
        ]
        action_batch = torch.cat(action_list, dim=0)

        outputs = self._run_inference(
            lambda: self._network.recurrent_inference(hidden_batch, action_batch)
        )

        for i, req in enumerate(self._recurrent_queue):
            single = NetworkOutput(
                value=outputs.value[i],
                reward=outputs.reward[i],
                policy_logits=outputs.policy_logits[i],
                hidden_state=outputs.hidden_state[i],
                value_prefix=(
                    outputs.value_prefix[i] if outputs.value_prefix is not None else None
                ),
                projection=(outputs.projection[i] if outputs.projection is not None else None),
            )
            req.callback(single)

        self._recurrent_queue.clear()

    def flush(self) -> None:
        """Process all queued requests (both initial and recurrent)."""
        self.flush_initial()
        self.flush_recurrent()

    def _run_inference(self, fn: Callable[[], NetworkOutput]) -> NetworkOutput:
        """Run network inference with optional AMP context.

        Args:
            fn: Inference function to call.

        Returns:
            NetworkOutput from the network.
        """
        with torch.no_grad():
            if self._use_amp:
                with torch.amp.autocast("cuda", dtype=self._amp_dtype):
                    return fn()
            return fn()
