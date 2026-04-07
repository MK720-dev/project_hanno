"""
Observation construction for Hanno / HannoNet1.

Role in the framework
---------------------
This file turns the environment's internal optimizee-side state and diagnostic
history into a compact observation vector that is sent to the controller.

This is intentionally different from the full underlying state:
- the state is the full condition of the optimization process,
- the observation is the controller-facing summary of that condition.

For the restricted proof of concept, we deliberately keep the observation
low-dimensional and interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hanno.environment.diagnostics import DiagnosticsTrace


@dataclass
class ObservationConfig:
    """Configuration for observation construction."""

    window: int = 3
    eps: float = 1e-8


class ObservationBuilder:
    """
    Build compact controller-facing observations from diagnostics.

    Current observation layout:
    0. log(current loss + eps)
    1. log(previous loss + eps)
    2. log(window-start loss + eps)
    3. one-step log improvement
    4. windowed log improvement
    5. latest gradient norm
    6. latest update norm
    7. latest effective learning rate
    8. step fraction in [0, 1]
    9. latest instability flag as 0/1
    """

    def __init__(self, config: ObservationConfig | None = None) -> None:
        self.config = config or ObservationConfig()

    def build(
        self,
        trace: DiagnosticsTrace,
        step_index: int,
        horizon: int,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Construct the observation tensor for the controller."""

        eps = self.config.eps

        current_loss = trace.latest_loss()
        if current_loss is None:
            current_loss = 0.0

        previous_loss = trace.previous_loss()
        if previous_loss is None:
            previous_loss = current_loss

        window_start_loss = trace.window_start_loss(self.config.window)
        if window_start_loss is None:
            window_start_loss = current_loss

        # Log-loss features align naturally with the chosen reward family.
        log_current = torch.log(torch.tensor(current_loss + eps, dtype=torch.float32))
        log_previous = torch.log(torch.tensor(previous_loss + eps, dtype=torch.float32))
        log_window_start = torch.log(
            torch.tensor(window_start_loss + eps, dtype=torch.float32)
        )

        one_step_improvement = log_previous - log_current
        windowed_improvement = log_window_start - log_current

        grad_norm = trace.grad_norms[-1] if trace.grad_norms else 0.0
        update_norm = trace.update_norms[-1] if trace.update_norms else 0.0
        effective_lr = trace.effective_lrs[-1] if trace.effective_lrs else 0.0
        instability_flag = 1.0 if (trace.instability_flags[-1] if trace.instability_flags else False) else 0.0

        step_fraction = float(step_index) / float(max(horizon, 1))

        return torch.tensor(
            [
                float(log_current.item()),
                float(log_previous.item()),
                float(log_window_start.item()),
                float(one_step_improvement.item()),
                float(windowed_improvement.item()),
                float(grad_norm),
                float(update_norm),
                float(effective_lr),
                float(step_fraction),
                float(instability_flag),
            ],
            dtype=torch.float32,
            device=device,
        )
