"""
Main Gym-like training environment for Hanno / HannoNet1.

Role in the framework
---------------------
This file implements the step-by-step interaction loop between:
- the optimizee task,
- the update engine,
- the reward function,
- the diagnostics trace,
- and the controller-facing observation builder.

Important conceptual boundary
-----------------------------
This environment does not train the controller.
It only:
- runs the inner optimizee dynamics,
- consumes controller-produced control,
- computes reward,
- and returns the next observation.

Controller training via REINFORCE/PPO belongs to the outer RL training layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from hanno.core.types import ControlOutput
from hanno.environment.diagnostics import DiagnosticsTrace, compute_step_diagnostics
from hanno.environment.observation import ObservationBuilder
from hanno.environment.reward import BaseReward, RewardInfo
from hanno.environment.update_engine import UpdateEngine
from hanno.tasks.base import BaseTask


@dataclass
class EnvStepResult:
    """Structured result returned by one environment step."""

    observation: torch.Tensor
    reward: float
    done: bool
    info: dict[str, Any]


class TrainingEnv:
    """
    Gym-like environment for one controlled optimizee episode.

    This follows the restricted HannoNet1 loop:
    controller -> control -> update engine -> optimizee/loss ->
    reward + diagnostics -> next observation
    """

    def __init__(
        self,
        task: BaseTask,
        update_engine: UpdateEngine,
        reward_fn: BaseReward,
        observation_builder: ObservationBuilder,
        instability_threshold: float = 1.5,
    ) -> None:
        self.task = task
        self.update_engine = update_engine
        self.reward_fn = reward_fn
        self.observation_builder = observation_builder
        self.instability_threshold = float(instability_threshold)

        self.task_state = None
        self.trace = DiagnosticsTrace()
        self.done = False

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Reset the environment to a fresh optimizee episode.

        The reset:
        1. creates a fresh task state,
        2. binds a fresh optimizer to the new optimizee parameters,
        3. computes the initial loss,
        4. initializes the diagnostics trace,
        5. returns the initial controller observation.
        """

        self.task_state = self.task.reset(seed=seed)
        self.update_engine.bind(self.task_state)
        self.trace = DiagnosticsTrace()
        self.done = False

        initial_loss = self.task.compute_loss(self.task_state)

        # At reset time, there is no gradient/update yet, so zeros are used.
        initial_diag = compute_step_diagnostics(
            loss=initial_loss,
            parameters=self.task_state.parameters.detach(),
            gradient=None,
            update=None,
            lr_effective=self.update_engine.base_lr,
            previous_loss=None,
            instability_threshold=self.instability_threshold,
        )
        self.trace.append(initial_diag)

        obs = self.observation_builder.build(
            trace=self.trace,
            step_index=self.task_state.step_index,
            horizon=self.task.horizon,
            device=self.task_state.parameters.device,
        )

        info = {
            "task_name": self.task.name,
            "task_metadata": dict(self.task_state.metadata),
            "initial_loss": initial_diag.loss,
        }

        return obs, info

    def step(self, control: ControlOutput) -> tuple[torch.Tensor, float, bool, dict[str, Any]]:
        """
        Apply one controller-guided optimizee step.

        The environment step updates the optimizee, not the controller.
        The controller is trained later by the outer RL algorithm using
        rewards collected from repeated calls to this method.
        """

        if self.done:
            raise RuntimeError(
                "Cannot call step() on a finished episode. Call reset() first."
            )

        if self.task_state is None:
            raise RuntimeError(
                "Environment has no task state. Call reset() before step()."
            )

        def loss_closure() -> torch.Tensor:
            return self.task.compute_loss(self.task_state)

        previous_loss = self.trace.latest_loss()

        # Inner-loop optimizee update. This is where the update engine applies
        # the controller-guided step to the optimizee.
        update_result = self.update_engine.step(
            task_state=self.task_state,
            control=control,
            loss_closure=loss_closure,
        )

        self.task_state.step_index += 1

        # Recompute the post-step optimizee loss so the environment can define
        # the new state, reward, and diagnostics.
        post_step_loss = self.task.compute_loss(self.task_state).detach()

        step_diag = compute_step_diagnostics(
            loss=post_step_loss,
            parameters=self.task_state.parameters.detach(),
            gradient=update_result.gradient,
            update=update_result.update_vector,
            lr_effective=update_result.lr_effective,
            previous_loss=previous_loss,
            instability_threshold=self.instability_threshold,
        )
        self.trace.append(step_diag)

        reward_info: RewardInfo = self.reward_fn.compute(self.trace)

        next_observation = self.observation_builder.build(
            trace=self.trace,
            step_index=self.task_state.step_index,
            horizon=self.task.horizon,
            device=self.task_state.parameters.device,
        )

        nonfinite_loss = not torch.isfinite(post_step_loss).item()
        reached_horizon = self.task_state.step_index >= self.task.horizon
        self.done = bool(nonfinite_loss or reached_horizon)

        info = {
            "task_name": self.task.name,
            "step_index": self.task_state.step_index,
            "loss": step_diag.loss,
            "grad_norm": step_diag.grad_norm,
            "update_norm": step_diag.update_norm,
            "effective_lr": step_diag.lr_effective,
            "instability_flag": float(step_diag.instability_flag),
            "reward_components": dict(reward_info.components),
            "raw_action": (
                control.raw_action.detach().cpu().tolist()
                if control.raw_action is not None
                else None
            ),
        }

        return next_observation, reward_info.reward, self.done, info
