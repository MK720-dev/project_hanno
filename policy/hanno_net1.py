"""
HannoNet1 controller definition.

Role in the framework
---------------------
This file defines the first controller model used in the restricted proof of
concept. HannoNet1 is a 2-layer LSTM controller followed by a Gaussian scalar
action head.

Conceptual role
---------------
The controller does not directly update the optimizee. Instead:
- it reads the environment observation,
- it outputs a stochastic action,
- the action is mapped into optimizer-level control,
- and the update engine uses that control to update the optimizee.

So HannoNet1 is a controller over training dynamics, not a replacement for
gradient descent itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hanno.policy.action_head import ActionMapper, GaussianScalarActionHead


@dataclass
class PolicyStepOutput:
    """
    Structured output of one controller step.
    """

    raw_action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    hidden: tuple[torch.Tensor, torch.Tensor]


class HannoNet1(nn.Module):
    """
    Two-layer LSTM controller for the restricted Hanno proof of concept.
    """

    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        min_lr_multiplier: float = 0.1,
        max_lr_multiplier: float = 3.0,
    ) -> None:
        super().__init__()

        self.observation_dim = int(observation_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        self.lstm = nn.LSTM(
            input_size=self.observation_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.action_head = GaussianScalarActionHead(hidden_dim=self.hidden_dim)
        self.action_mapper = ActionMapper(
            min_lr_multiplier=min_lr_multiplier,
            max_lr_multiplier=max_lr_multiplier,
        )

    def init_hidden(
        self,
        batch_size: int = 1,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create an all-zero initial LSTM hidden state.
        """

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return h0, c0

    def forward_step(
        self,
        observation: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> PolicyStepOutput:
        """
        Run one controller step on a single observation.
        """

        if observation.dim() == 1:
            obs = observation.unsqueeze(0).unsqueeze(0)
        elif observation.dim() == 2:
            obs = observation.unsqueeze(1)
        else:
            raise ValueError(
                "Expected observation with shape [obs_dim] or [batch, obs_dim]."
            )

        lstm_out, next_hidden = self.lstm(obs, hidden)
        features = lstm_out[:, -1, :]

        head_output = self.action_head(features)
        distribution = head_output.distribution

        raw_action = distribution.rsample()
        log_prob = distribution.log_prob(raw_action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)

        return PolicyStepOutput(
            raw_action=raw_action,
            log_prob=log_prob,
            entropy=entropy,
            mean=head_output.mean,
            std=head_output.std,
            hidden=next_hidden,
        )
