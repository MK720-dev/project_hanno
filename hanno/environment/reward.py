"""
Reward functions for Hanno / HannoNet1.

Role in the framework
---------------------
This file contains reward logic used by the environment to score optimizee-side
behavior. These rewards are based on the optimizee loss trajectory, not the
controller's policy-gradient loss.

This distinction matters:
- optimizee loss lives in the inner optimization process,
- controller loss lives in the outer RL update process.

The first proof-of-concept reward family follows the current project notes:
- plain log-improvement for debugging/sanity checks,
- windowed log-improvement for smoother progress measurement,
- windowed log-improvement with a simple instability penalty for main runs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from hanno.environment.diagnostics import DiagnosticsTrace


@dataclass
class RewardInfo:
    """Structured reward output with scalar reward and component breakdown."""

    reward: float
    components: dict[str, float]


class BaseReward:
    """Lightweight base class for reward functions."""

    def compute(self, trace: DiagnosticsTrace) -> RewardInfo:
        raise NotImplementedError


class LogImprovementReward(BaseReward):
    """
    Plain one-step log-improvement reward.

    Formula:
        r_t = log(l_{t-1} + eps) - log(l_t + eps)
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = float(eps)

    def compute(self, trace: DiagnosticsTrace) -> RewardInfo:
        if len(trace.losses) < 2:
            return RewardInfo(reward=0.0, components={"log_improvement": 0.0})

        prev_loss = trace.losses[-2]
        curr_loss = trace.losses[-1]
        value = math.log(prev_loss + self.eps) - math.log(curr_loss + self.eps)

        return RewardInfo(
            reward=value,
            components={"log_improvement": value},
        )


class WindowedLogImprovementReward(BaseReward):
    """
    Windowed log-improvement reward.

    Formula:
        r_t = log(l_{t-k} + eps) - log(l_t + eps)
    """

    def __init__(self, window: int = 3, eps: float = 1e-8) -> None:
        self.window = int(window)
        self.eps = float(eps)

    def compute(self, trace: DiagnosticsTrace) -> RewardInfo:
        if not trace.losses:
            return RewardInfo(reward=0.0, components={"windowed_log_improvement": 0.0})

        start_loss = trace.window_start_loss(self.window)
        curr_loss = trace.losses[-1]

        if start_loss is None:
            return RewardInfo(reward=0.0, components={"windowed_log_improvement": 0.0})

        value = math.log(start_loss + self.eps) - math.log(curr_loss + self.eps)

        return RewardInfo(
            reward=value,
            components={"windowed_log_improvement": value},
        )


class WindowedLogImprovementWithInstabilityPenalty(BaseReward):
    """
    Windowed log-improvement with a simple instability penalty.

    Formula:
        r_t = [log(l_{t-k} + eps) - log(l_t + eps)] - lambda * 1_{l_t > c l_{t-k}}
    """

    def __init__(
        self,
        window: int = 3,
        eps: float = 1e-8,
        penalty_weight: float = 1.0,
        instability_threshold: float = 1.5,
    ) -> None:
        self.window = int(window)
        self.eps = float(eps)
        self.penalty_weight = float(penalty_weight)
        self.instability_threshold = float(instability_threshold)

    def compute(self, trace: DiagnosticsTrace) -> RewardInfo:
        if not trace.losses:
            return RewardInfo(
                reward=0.0,
                components={
                    "windowed_log_improvement": 0.0,
                    "instability_penalty": 0.0,
                },
            )

        start_loss = trace.window_start_loss(self.window)
        curr_loss = trace.losses[-1]

        if start_loss is None:
            return RewardInfo(
                reward=0.0,
                components={
                    "windowed_log_improvement": 0.0,
                    "instability_penalty": 0.0,
                },
            )

        progress = math.log(start_loss + self.eps) - math.log(curr_loss + self.eps)

        penalty = 0.0
        if start_loss > 0.0 and curr_loss > self.instability_threshold * start_loss:
            penalty = self.penalty_weight

        total = progress - penalty

        return RewardInfo(
            reward=total,
            components={
                "windowed_log_improvement": progress,
                "instability_penalty": penalty,
            },
        )
