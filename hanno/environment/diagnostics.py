"""
Diagnostics helpers for Hanno optimization episodes.

This file collects compact, interpretable optimizee-side summaries such as:
- scalar loss,
- gradient norm,
- parameter norm,
- update norm,
- effective learning rate,
- instability flag.

MNIST extension note
--------------------
The analytical-only prototype assumed a single optimizee parameter tensor. This
version computes norms generically for either:
- a single Parameter, or
- a full nn.Module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from hanno.core.types import StepDiagnostics
from hanno.core.utils import flatten_parameters


@dataclass
class DiagnosticsTrace:
    """
    Rolling trace of recent optimization statistics.
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
        """Return the previous loss if available."""
        return self.losses[-2] if len(self.losses) >= 2 else None

    def window_start_loss(self, window: int) -> float | None:
        """
        Return the loss at t-k where k = window, clipped to the earliest
        available element at the start of a rollout.
        """

        if not self.losses:
            return None
        index = max(0, len(self.losses) - 1 - window)
        return self.losses[index]


def tensor_l2_norm(tensor: torch.Tensor | None) -> float:
    """
    Return the Euclidean norm of a tensor as a Python float.
    """

    if tensor is None:
        return 0.0
    return float(torch.norm(tensor.detach()).item())


def compute_step_diagnostics(
    *,
    loss: torch.Tensor,
    parameters: nn.Module | nn.Parameter,
    gradient: torch.Tensor | None,
    update: torch.Tensor | None,
    lr_effective: float,
    previous_loss: float | None,
    instability_threshold: float = 1.5,
) -> StepDiagnostics:
    """
    Build a StepDiagnostics object from raw optimizee quantities.
    """

    loss_value = float(loss.detach().item())
    grad_norm = tensor_l2_norm(gradient)
    param_norm = tensor_l2_norm(flatten_parameters(parameters))
    update_norm = tensor_l2_norm(update)

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
