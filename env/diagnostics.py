"""
Diagnostics helpers for Hanno optimization episodes.

The restricted proof-of-concept implementation relies heavily on compact,
interpretable diagnostics rather than raw parameter/state exposure. This file
provides helper functions for computing the quantities that later become part
of observations, reward logic, logging, and debugging.

The focus here is on small numerical summaries such as:
- scalar loss,
- gradient norm,
- parameter norm,
- update norm,
- and a simple instability indicator.

These summaries align well with the current project direction, where early
experiments should remain traceable and easy to interpret.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch

from hanno.core.types import StepDiagnostics


@dataclass
class DiagnosticsTrace:
    """
    Rolling trace of recent optimization statistics.

    The environment can keep one of these per episode and update it after each
    optimization step. A short rolling history is especially useful for
    building observations and windowed rewards later.
    """

    losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    param_norms: list[float] = field(default_factory=list)
    update_norms: list[float] = field(default_factory=list)
    effective_lrs: list[float] = field(default_factory=list)
    instability_flags: list[bool] = field(default_factory=list)

    def append(self, diagnostics: StepDiagnostics) -> None:
        """Append one new step's diagnostics to the rolling trace."""
        self.losses.append(diagnostics.loss)
        self.grad_norms.append(diagnostics.grad_norm)
        self.param_norms.append(diagnostics.param_norm)
        self.update_norms.append(diagnostics.update_norm)
        self.effective_lrs.append(diagnostics.lr_effective)
        self.instability_flags.append(diagnostics.instability_flag)

    def latest_loss(self) -> float | None:
        """Return the latest loss if available."""
        return self.losses[-1] if self.losses else None

    def previous_loss(self) -> float | None:
        """Return the previous loss if at least two losses have been recorded."""
        return self.losses[-2] if len(self.losses) >= 2 else None

    def window_start_loss(self, window: int) -> float | None:
        """
        Return the loss at the beginning of a given lookback window.

        If the trace is shorter than the requested window, the earliest
        available loss is returned instead.
        """

        if not self.losses:
            return None
        index = max(0, len(self.losses) - 1 - window)
        return self.losses[index]


def tensor_l2_norm(tensor: torch.Tensor | None) -> float:
    """
    Return the Euclidean norm of a tensor as a Python float.

    Returning a plain float makes the value easy to log and store.
    """

    if tensor is None:
        return 0.0
    return float(torch.norm(tensor.detach()).item())


def compute_step_diagnostics(
    *,
    loss: torch.Tensor,
    parameters: torch.Tensor,
    gradient: torch.Tensor | None,
    update: torch.Tensor | None,
    lr_effective: float,
    previous_loss: float | None,
    instability_threshold: float = 1.5,
) -> StepDiagnostics:
    """
    Build a StepDiagnostics object from raw optimization quantities.

    Parameters
    ----------
    loss:
        Current scalar loss tensor.
    parameters:
        Current optimizee parameters after the step.
    gradient:
        Gradient tensor evaluated for the current step.
    update:
        Applied update vector, if known. In some early debugging scenarios this
        may be unavailable, in which case the norm is recorded as zero.
    lr_effective:
        Effective learning rate used by the update engine after modulation.
    previous_loss:
        Previous step's scalar loss, used to define a simple instability flag.
    instability_threshold:
        Threshold c used in simple spike detection: loss_t > c * loss_{t-1}.
    """

    loss_value = float(loss.detach().item())
    grad_norm = tensor_l2_norm(gradient)
    param_norm = tensor_l2_norm(parameters)
    update_norm = tensor_l2_norm(update)

    # A minimal instability diagnostic: treat a sufficiently large upward spike
    # in the loss as instability. This aligns with the simple penalty ideas
    # discussed in the current project notes.
    instability_flag = False
    if previous_loss is not None and previous_loss > 0.0:
        instability_flag = loss_value > instability_threshold * previous_loss

    return StepDiagnostics(
        loss=loss_value,
        grad_norm=grad_norm,
        param_norm=param_norm,
        update_norm=update_norm,
        lr_effective=float(lr_effective),
        instability_flag=instability_flag,
        extras={},
    )
