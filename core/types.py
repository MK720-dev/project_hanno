"""
Shared dataclasses and typed containers for Hanno / HannoNet1.

This file defines the small set of typed objects that different parts of the
framework exchange with one another. The main design goal is to make the code
interfaces explicit early, so the restricted proof-of-concept implementation
stays clean and extensible.

The current project direction, based on the restricted schema, is that:
- the controller produces optimizer-level control,
- the update engine consumes that control,
- the environment produces diagnostics and rewards,
- and the RL trainer stores trajectories over full episodes.

Even though HannoNet1 will initially only control the learning rate multiplier,
we still keep a structured OptimizerAction object inside a broader
ControlOutput object. That preserves the conceptual separation between:
- optimizer-level action a_t, and
- future preconditioning / update-modulation signal p_t.

This small decision keeps the first version simple without forcing a later
rewrite when the controller outputs become richer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Controller-side control objects
# ---------------------------------------------------------------------------

@dataclass
class OptimizerAction:
    """
    Optimizer-level control produced by the policy at one time step.

    For HannoNet1, the only active control is the learning-rate multiplier.
    Later versions can extend this object with additional optimizer-level
    signals such as momentum scaling, weight decay scaling, clipping control,
    or mode-selection variables.

    Attributes
    ----------
    lr_multiplier:
        A positive multiplier applied to the optimizer backbone's base learning
        rate. For example, if the base learning rate is 1e-2 and the
        lr_multiplier is 0.5, then the effective learning rate becomes 5e-3.
    """

    lr_multiplier: float


@dataclass
class ControlOutput:
    """
    Full control payload passed from the policy to the environment/update engine.

    The structured shape matters even in the restricted proof-of-concept setup.
    The grand schema distinguishes optimizer-level action a_t from a future
    update-modulation / preconditioning signal p_t, so we preserve that split
    in the software interface from the beginning.

    Attributes
    ----------
    optimizer_action:
        The optimizer-level decisions made by the controller at the current
        step. In HannoNet1 this mainly means learning-rate modulation.
    preconditioner_signal:
        Reserved for future versions where the controller may directly reshape
        the update direction or magnitude in a more geometric way.
    raw_action:
        The unsquashed or minimally processed tensor sampled from the policy
        distribution. Keeping it is helpful for debugging, logging, and later
        action regularization experiments.
    """

    optimizer_action: OptimizerAction
    preconditioner_signal: torch.Tensor | None = None
    raw_action: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Task and environment state containers
# ---------------------------------------------------------------------------

@dataclass
class TaskState:
    """
    State returned when a task is reset.

    This object keeps the optimizee-side state in one place so the environment
    can evolve it during an episode. In the analytical-task phase, the key
    parameter is a learnable vector x. Later, this container can hold a neural
    network model, optimizer objects, datasets, or task-specific metadata.

    Attributes
    ----------
    parameters:
        The optimizee parameters currently being trained or optimized.
    step_index:
        Current time step inside the episode.
    metadata:
        Free-form task-specific information that is useful for logging or
        debugging. Examples include the task name, dimension, or curvature info.
    """

    parameters: torch.nn.Parameter
    step_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepDiagnostics:
    """
    Diagnostics collected for a single environment step.

    These values are intentionally low-dimensional and interpretable, because
    the first version of the project relies on compact diagnostic observations
    rather than raw model weights.

    Attributes
    ----------
    loss:
        Scalar loss value after the step.
    grad_norm:
        Euclidean norm of the gradient at the step.
    param_norm:
        Euclidean norm of the current parameter vector.
    update_norm:
        Euclidean norm of the applied parameter update, if available.
    lr_effective:
        Effective learning rate after policy modulation.
    instability_flag:
        Binary indicator used by simple stability-aware reward variants.
    extras:
        Additional task- or experiment-specific values.
    """

    loss: float
    grad_norm: float
    param_norm: float
    update_norm: float
    lr_effective: float
    instability_flag: bool
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeStep:
    """
    One transition entry stored in a rollout trajectory.

    This object keeps together the core information needed by policy-gradient
    methods such as REINFORCE: state, action, log-probability, reward, and
    diagnostics. Keeping it explicit now makes later debugging much easier.
    """

    observation: torch.Tensor
    control: ControlOutput
    reward: float
    log_prob: torch.Tensor
    diagnostics: StepDiagnostics


@dataclass
class EpisodeTrajectory:
    """
    Full trajectory collected over one optimization episode.

    The first RL implementation will likely use a simple list of EpisodeStep
    entries and compute returns after the episode ends.
    """

    steps: list[EpisodeStep] = field(default_factory=list)

    def append(self, step: EpisodeStep) -> None:
        """Append one step to the trajectory."""
        self.steps.append(step)

    def rewards(self) -> list[float]:
        """Return the scalar rewards from all stored steps."""
        return [step.reward for step in self.steps]

    def __len__(self) -> int:
        """Return the number of steps collected in the episode."""
        return len(self.steps)
