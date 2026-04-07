"""
REINFORCE trainer for Hanno / HannoNet1.

Role in the framework
---------------------
This file implements the outer-loop controller update rule. Unlike the update
engine, which updates the optimizee every environment step, REINFORCE updates
the controller parameters after a rollout has been collected.

Conceptual split
----------------
- inner loop: optimizee updated by update engine
- outer loop: controller updated by REINFORCE from rollout rewards
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hanno.core.types import EpisodeTrajectory


@dataclass
class ReinforceUpdateStats:
    """
    Structured summary of one REINFORCE parameter update.
    """

    policy_loss: float
    entropy_bonus: float
    total_loss: float
    mean_return: float


class REINFORCETrainer:
    """
    Minimal REINFORCE trainer for HannoNet1.
    """

    def __init__(
        self,
        policy: nn.Module,
        learning_rate: float = 1e-3,
        gamma: float = 1.0,
        entropy_weight: float = 1e-3,
        normalize_returns: bool = True,
        grad_clip_norm: float | None = 1.0,
    ) -> None:
        self.policy = policy
        self.gamma = float(gamma)
        self.entropy_weight = float(entropy_weight)
        self.normalize_returns = bool(normalize_returns)
        self.grad_clip_norm = grad_clip_norm

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

    def compute_discounted_returns(self, rewards: list[float]) -> torch.Tensor:
        """
        Compute discounted Monte Carlo returns from a reward list.
        """

        returns = []
        running = 0.0

        for reward in reversed(rewards):
            running = reward + self.gamma * running
            returns.append(running)

        returns.reverse()
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        if self.normalize_returns and len(returns_tensor) > 1:
            mean = returns_tensor.mean()
            std = returns_tensor.std(unbiased=False)
            returns_tensor = (returns_tensor - mean) / (std + 1e-8)

        return returns_tensor

    def update(self, trajectory: EpisodeTrajectory) -> ReinforceUpdateStats:
        """
        Apply one REINFORCE update from a collected rollout trajectory.
        """

        if len(trajectory) == 0:
            raise ValueError("Cannot update from an empty trajectory.")

        returns = self.compute_discounted_returns(trajectory.rewards())

        log_probs = torch.stack([step.log_prob.squeeze() for step in trajectory.steps])
        entropies = torch.stack(
            [step.diagnostics.extras["entropy_tensor"].squeeze() for step in trajectory.steps]
        )

        policy_loss_tensor = -(log_probs * returns.to(log_probs.device)).sum()
        entropy_bonus_tensor = entropies.sum()
        total_loss_tensor = policy_loss_tensor - self.entropy_weight * entropy_bonus_tensor

        self.optimizer.zero_grad()
        total_loss_tensor.backward()

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)

        self.optimizer.step()

        return ReinforceUpdateStats(
            policy_loss=float(policy_loss_tensor.detach().item()),
            entropy_bonus=float(entropy_bonus_tensor.detach().item()),
            total_loss=float(total_loss_tensor.detach().item()),
            mean_return=float(returns.mean().detach().item()),
        )
