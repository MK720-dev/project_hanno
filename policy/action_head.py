"""
Action-head utilities for Hanno / HannoNet1.

Role in the framework
---------------------
This file translates controller-side latent outputs into:
- a stochastic action distribution,
- a sampled scalar action,
- a bounded learning-rate multiplier,
- and finally a structured ControlOutput object.

This separation keeps three logically distinct steps apart:
1. the policy network produces features,
2. the action head defines the stochastic policy distribution,
3. the action mapper turns sampled actions into environment-facing controls.

Current restricted-scope meaning
--------------------------------
In HannoNet1, the sampled action is a single scalar. It is not an optimizee
update by itself. Instead, it is mapped into an optimizer-level control:
a bounded learning-rate multiplier inside OptimizerAction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal

from hanno.core.types import ControlOutput, OptimizerAction


@dataclass
class ActionHeadOutput:
    """
    Structured output from the Gaussian scalar action head.
    """

    mean: torch.Tensor
    std: torch.Tensor
    distribution: Normal


class GaussianScalarActionHead(nn.Module):
    """
    Gaussian scalar action head for continuous control.

    The controller produces a hidden feature vector. This head converts that
    feature vector into:
    - a scalar mean,
    - a scalar standard deviation,
    - and a 1D Gaussian action distribution.

    A stochastic policy is required for REINFORCE.
    """

    def __init__(
        self,
        hidden_dim: int,
        init_log_std: float = -0.5,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        self.mean_layer = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.tensor([init_log_std], dtype=torch.float32))
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

    def forward(self, features: torch.Tensor) -> ActionHeadOutput:
        """
        Build the Gaussian action distribution from controller features.
        """

        mean = self.mean_layer(features)
        log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std).expand_as(mean)
        distribution = Normal(mean, std)
        return ActionHeadOutput(mean=mean, std=std, distribution=distribution)


class ActionMapper:
    """
    Map sampled controller actions into bounded optimizer-level controls.

    Current restricted mapping:
        raw scalar action z_t
            -> tanh(z_t) in [-1, 1]
            -> affine map into [min_lr_multiplier, max_lr_multiplier]
            -> pack into OptimizerAction / ControlOutput
    """

    def __init__(
        self,
        min_lr_multiplier: float = 0.1,
        max_lr_multiplier: float = 3.0,
    ) -> None:
        if min_lr_multiplier <= 0.0:
            raise ValueError("min_lr_multiplier must be positive.")
        if max_lr_multiplier <= min_lr_multiplier:
            raise ValueError("max_lr_multiplier must be greater than min_lr_multiplier.")

        self.min_lr_multiplier = float(min_lr_multiplier)
        self.max_lr_multiplier = float(max_lr_multiplier)

    def raw_action_to_lr_multiplier(self, raw_action: torch.Tensor) -> torch.Tensor:
        """
        Convert a raw scalar action into a bounded positive LR multiplier.
        """

        squashed = torch.tanh(raw_action)
        normalized = 0.5 * (squashed + 1.0)
        lr_multiplier = (
            self.min_lr_multiplier
            + normalized * (self.max_lr_multiplier - self.min_lr_multiplier)
        )
        return lr_multiplier

    def to_control_output(self, raw_action: torch.Tensor) -> ControlOutput:
        """
        Build the environment-facing ControlOutput from a sampled raw action.
        """

        lr_multiplier_tensor = self.raw_action_to_lr_multiplier(raw_action)
        lr_multiplier = float(lr_multiplier_tensor.squeeze().detach().item())

        optimizer_action = OptimizerAction(lr_multiplier=lr_multiplier)

        return ControlOutput(
            optimizer_action=optimizer_action,
            preconditioner_signal=None,
            raw_action=raw_action.detach(),
        )
