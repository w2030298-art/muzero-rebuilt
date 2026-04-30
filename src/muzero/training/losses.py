"""MuZero loss functions: policy cross-entropy, value MSE, reward MSE."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from muzero.core.types import TargetSequence
from muzero.models.outputs import NetworkOutput


@dataclass(slots=True)
class LossBreakdown:
    """Breakdown of individual loss components."""

    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    value_prefix: torch.Tensor | None = None
    consistency: torch.Tensor | None = None


class MuZeroLoss:
    """Standard MuZero loss: policy + value + reward."""

    def policy_loss(
        self,
        pred_policy_logits: torch.Tensor,
        target_policy: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-entropy loss between predicted logits and soft targets.

        Args:
            pred_policy_logits: shape ``[B, action_dim]``.
            target_policy: soft policy targets, shape ``[B, action_dim]``.
            mask: optional mask for valid positions, shape ``[B]``.

        Returns:
            Scalar loss.
        """
        log_probs = F.log_softmax(pred_policy_logits, dim=-1)
        loss = -(target_policy * log_probs).sum(dim=-1)
        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1)
        return loss.mean()

    def value_loss(self, pred_value: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
        """MSE loss for value predictions.

        Args:
            pred_value: Predicted values.
            target_value: Target values.

        Returns:
            Scalar loss.
        """
        return F.mse_loss(pred_value, target_value)

    def reward_loss(self, pred_reward: torch.Tensor, target_reward: torch.Tensor) -> torch.Tensor:
        """MSE loss for reward predictions.

        Args:
            pred_reward: Predicted rewards.
            target_reward: Target rewards.

        Returns:
            Scalar loss.
        """
        return F.mse_loss(pred_reward, target_reward)

    def total_loss(
        self,
        predictions: list[NetworkOutput],
        targets: TargetSequence,
    ) -> LossBreakdown:
        """Compute total MuZero loss over unrolled predictions.

        Args:
            predictions: List of network outputs for each unroll step (len=K).
            targets: TargetSequence with value, reward, policy targets.

        Returns:
            LossBreakdown with total and component losses.
        """
        K = len(predictions)
        policy_losses = []
        value_losses = []
        reward_losses = []

        for k in range(K):
            pred = predictions[k]
            # Policy loss (from initial step's output)
            if k == 0:
                policy_losses.append(
                    self.policy_loss(pred.policy_logits, targets.target_policies[k])
                )
            value_losses.append(self.value_loss(pred.value, targets.target_values[k]))
            reward_losses.append(self.reward_loss(pred.reward, targets.target_rewards[k]))

        loss_p = torch.stack(policy_losses).mean()
        loss_v = torch.stack(value_losses).mean()
        loss_r = torch.stack(reward_losses).mean()

        total = loss_p + loss_v + loss_r

        return LossBreakdown(
            total=total,
            policy=loss_p.detach(),
            value=loss_v.detach(),
            reward=loss_r.detach(),
        )


class EfficientZeroLoss:
    """EfficientZero-specific loss functions: value prefix and consistency."""

    @staticmethod
    def value_prefix_loss(
        pred_value_prefix: torch.Tensor, target_value_prefix: torch.Tensor
    ) -> torch.Tensor:
        """MSE loss for value prefix predictions."""
        return F.mse_loss(pred_value_prefix, target_value_prefix)

    @staticmethod
    def consistency_loss(projection: torch.Tensor, target_projection: torch.Tensor) -> torch.Tensor:
        """Consistency loss using negative cosine similarity.

        Both inputs should be L2-normalized. Target is detached.
        """
        target = target_projection.detach()
        cos_sim = (projection * target).sum(dim=-1)
        return 2.0 - 2.0 * cos_sim.mean()
