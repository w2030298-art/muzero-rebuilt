"""Standard MuZero Trainer — training step, loop, and AMP support."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from muzero.config.schema import MuZeroConfig
from muzero.core.types import TrainingBatch
from muzero.models.base import BaseMuZeroNetwork
from muzero.models.outputs import NetworkOutput
from muzero.training.losses import LossBreakdown, MuZeroLoss


@dataclass(slots=True)
class TrainStepResult:
    """Result of a single training step."""

    step: int
    loss_total: float
    loss_policy: float
    loss_value: float
    loss_reward: float
    grad_norm: float


class Trainer:
    """Standard MuZero trainer with single-GPU training loop and AMP support.

    Args:
        network: The MuZero network to train.
        optimizer: PyTorch optimizer.
        loss_fn: MuZeroLoss instance.
        config: Full MuZero configuration.
        device: Torch device (cpu or cuda).
    """

    def __init__(
        self,
        network: BaseMuZeroNetwork,
        optimizer: torch.optim.Optimizer,
        loss_fn: MuZeroLoss,
        config: MuZeroConfig,
        device: torch.device,
    ) -> None:
        self._network = network
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._config = config
        self._device = device
        self._step = 0

        # AMP setup
        use_amp = config.training.precision != "fp32" and device.type == "cuda"
        self._amp_dtype = torch.float16
        if config.training.precision == "amp_bf16":
            self._amp_dtype = torch.bfloat16
        self._use_amp = use_amp
        self._scaler: torch.amp.GradScaler | None = None
        if use_amp and config.training.precision == "amp_fp16":
            self._scaler = torch.amp.GradScaler("cuda")

    @property
    def step(self) -> int:
        """Current training step."""
        return self._step

    def train_step(self, batch: TrainingBatch) -> TrainStepResult:
        """Execute one training step.

        Args:
            batch: Training batch with observations, actions, targets.

        Returns:
            TrainStepResult with loss and gradient norm.
        """
        self._network.train()
        self._optimizer.zero_grad(set_to_none=True)

        # Move batch to device
        obs = torch.from_numpy(batch.observations).float().to(self._device)
        actions = torch.from_numpy(batch.actions).long().to(self._device)
        target_values = torch.from_numpy(batch.target_values).float().to(self._device)
        target_rewards = torch.from_numpy(batch.target_rewards).float().to(self._device)
        target_policies = torch.from_numpy(batch.target_policies).float().to(self._device)

        batch_size = obs.shape[0]

        # Unroll predictions
        predictions = self._unroll_predictions(
            obs=obs,
            actions=actions,
            target_values=target_values,
            target_rewards=target_rewards,
            target_policies=target_policies,
            batch_size=batch_size,
        )

        # Compute loss
        if self._use_amp:
            with torch.amp.autocast("cuda", dtype=self._amp_dtype):
                loss_breakdown = self._compute_loss(
                    predictions, target_values, target_rewards, target_policies, batch_size
                )

            if self._scaler is not None:
                self._scaler.scale(loss_breakdown.total).backward()
                self._scaler.unscale_(self._optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=10.0)
                self._scaler.step(self._optimizer)
                self._scaler.update()
            else:
                loss_breakdown.total.backward()
                grad_norm = nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=10.0)
                self._optimizer.step()
        else:
            loss_breakdown = self._compute_loss(
                predictions, target_values, target_rewards, target_policies, batch_size
            )
            loss_breakdown.total.backward()
            grad_norm = nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=10.0)
            self._optimizer.step()

        self._step += 1

        return TrainStepResult(
            step=self._step,
            loss_total=float(loss_breakdown.total.item()),
            loss_policy=float(loss_breakdown.policy.item()),
            loss_value=float(loss_breakdown.value.item()),
            loss_reward=float(loss_breakdown.reward.item()),
            grad_norm=float(grad_norm.item()),
        )

    def _unroll_predictions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        target_values: torch.Tensor,
        target_rewards: torch.Tensor,
        target_policies: torch.Tensor,
        batch_size: int,
    ) -> list[NetworkOutput]:
        """Unroll network predictions over the sequence.

        Returns:
            List of NetworkOutput for each unroll step.
        """
        K = actions.shape[1]
        predictions: list[NetworkOutput] = []

        # Initial inference
        with torch.no_grad():
            init_output = self._network.initial_inference(obs[:, 0])

        predictions.append(init_output)

        # Recurrent inference for each step
        hidden_state = init_output.hidden_state
        for k in range(K):
            output = self._network.recurrent_inference(hidden_state, actions[:, k])
            predictions.append(output)
            hidden_state = output.hidden_state

        return predictions

    def _compute_loss(
        self,
        predictions: list[NetworkOutput],
        target_values: torch.Tensor,
        target_rewards: torch.Tensor,
        target_policies: torch.Tensor,
        batch_size: int,
    ) -> LossBreakdown:
        """Compute loss from predictions and targets.

        Args:
            predictions: List of NetworkOutput for each step.
            target_values: Target values.
            target_rewards: Target rewards.
            target_policies: Target policies.
            batch_size: Batch size.

        Returns:
            LossBreakdown.
        """
        K = len(predictions) - 1  # Predictions has K+1 entries
        policy_losses = []
        value_losses = []
        reward_losses = []

        for k in range(K):
            pred = predictions[k]
            policy_losses.append(
                self._loss_fn.policy_loss(pred.policy_logits, target_policies[:, k])
            )
            value_losses.append(self._loss_fn.value_loss(pred.value, target_values[:, k]))
            reward_losses.append(self._loss_fn.reward_loss(pred.reward, target_rewards[:, k]))

        # Final value from last prediction
        value_losses.append(self._loss_fn.value_loss(predictions[-1].value, target_values[:, K]))

        loss_p = torch.stack(policy_losses).mean()
        loss_v = torch.stack(value_losses).mean()
        loss_r = torch.stack(reward_losses).mean()

        return LossBreakdown(
            total=loss_p + loss_v + loss_r,
            policy=loss_p.detach(),
            value=loss_v.detach(),
            reward=loss_r.detach(),
        )

    def sync_weights(self) -> dict[str, torch.Tensor]:
        """Return current model weights as CPU tensors.

        Returns:
            Dictionary of parameter names to CPU tensor copies.
        """
        return self._network.get_weights()
