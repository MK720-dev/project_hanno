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

MNIST extension note
--------------------
Some tasks, such as MNIST, need per-step dataset progression. To support that
without redesigning the whole interface, the environment now checks whether the
task exposes an optional `advance(task_state)` method and calls it after each
completed step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from hanno.core.types import ControlOutput
from hanno.core.utils import get_optimizee_device
from hanno.environment.diagnostics import DiagnosticsTrace, compute_step_diagnostics
from hanno.environment.observation import ObservationBuilder
from hanno.environment.reward import BaseReward, RewardInfo
from hanno.environment.update_engine import UpdateEngine
from hanno.tasks.base import BaseTask


@dataclass
class EnvStepResult:
    """
    Structured result returned by one environment step.
    """

    observation: torch.Tensor
    reward: float
    done: bool
    info: dict[str, Any]


class TrainingEnv:
    """
    Gym-like environment for one controlled optimizee episode.
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
        """

        self.task_state = self.task.reset(seed=seed)
        self.update_engine.bind(self.task_state)
        self.trace = DiagnosticsTrace()
        self.done = False

        initial_loss = self.task.compute_loss(self.task_state)

        initial_diag = compute_step_diagnostics(
            loss=initial_loss,
            parameters=self.task_state.parameters,
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
            device=get_optimizee_device(self.task_state.parameters),
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

        update_result = self.update_engine.step(
            task_state=self.task_state,
            control=control,
            loss_closure=loss_closure,
        )

        self.task_state.step_index += 1
        post_step_loss = self.task.compute_loss(self.task_state).detach()

        if not torch.isfinite(post_step_loss):
            self.done = True
            fallback_observation = torch.zeros(
                10,
                dtype=torch.float32,
                device=get_optimizee_device(self.task_state.parameters),
            )

            info = {
                "task_name": self.task.name,
                "step_index": self.task_state.step_index,
                "loss": float("inf"),
                "grad_norm": float("inf"),
                "update_norm": float("inf"),
                "effective_lr": update_result.lr_effective,
                "instability_flag": 1.0,
                "reward_components": {"nonfinite_termination": 1.0},
                "raw_action": (
                    control.raw_action.detach().cpu().tolist()
                    if control.raw_action is not None
                    else None
                ),
            }

            return fallback_observation, -10.0, True, info

        step_diag = compute_step_diagnostics(
            loss=post_step_loss,
            parameters=self.task_state.parameters,
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
            device=get_optimizee_device(self.task_state.parameters),
        )

        # Advance dataset-backed tasks to their next batch after the current
        # step has been fully evaluated on the current batch.
        if hasattr(self.task, "advance"):
            self.task.advance(self.task_state)

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
