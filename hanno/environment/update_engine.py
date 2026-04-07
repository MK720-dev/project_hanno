"""
Controlled optimizee update engine for Hanno / HannoNet1.

Role in the framework
---------------------
This file implements the inner-loop update mechanism that acts on the optimizee.
This remains intentionally separate from REINFORCE or any other controller-
training logic.

MNIST extension note
--------------------
The update engine now supports optimizees that are either:
- a single Parameter vector (analytical phase), or
- a full nn.Module (MNIST and later CIFAR).

The controller still only modulates the effective learning rate in HannoNet1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hanno.core.types import ControlOutput, TaskState
from hanno.core.utils import flatten_gradients, flatten_parameters, optimizee_parameters


@dataclass
class UpdateResult:
    """
    Result object returned after one controlled optimizee step.
    """

    loss: torch.Tensor
    gradient: torch.Tensor
    update_vector: torch.Tensor
    lr_effective: float


class UpdateEngine:
    """
    Fixed-backbone optimizee update engine with controller-driven modulation.
    """

    def __init__(
        self,
        optimizer_name: str = "sgd",
        base_lr: float = 1e-2,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
    ) -> None:
        self.optimizer_name = optimizer_name.lower()
        self.base_lr = float(base_lr)
        self.adam_betas = adam_betas
        self.adam_eps = adam_eps
        self._optimizer: torch.optim.Optimizer | None = None

    def bind(self, task_state: TaskState) -> None:
        """
        Create and attach a fresh backbone optimizer to the current optimizee.
        """

        params = optimizee_parameters(task_state.parameters)

        if self.optimizer_name == "sgd":
            self._optimizer = torch.optim.SGD(params, lr=self.base_lr)
        elif self.optimizer_name == "adam":
            self._optimizer = torch.optim.Adam(
                params,
                lr=self.base_lr,
                betas=self.adam_betas,
                eps=self.adam_eps,
            )
        else:
            raise ValueError(
                f"Unsupported optimizer_name='{self.optimizer_name}'. "
                "Expected 'sgd' or 'adam'."
            )

    def _require_optimizer(self) -> torch.optim.Optimizer:
        """
        Return the bound optimizer or raise an informative error.
        """

        if self._optimizer is None:
            raise RuntimeError(
                "UpdateEngine is not bound to a task state. "
                "Call bind(task_state) before stepping."
            )
        return self._optimizer

    def step(
        self,
        task_state: TaskState,
        control: ControlOutput,
        loss_closure,
    ) -> UpdateResult:
        """
        Apply one controlled optimizee update.
        """

        optimizer = self._require_optimizer()

        lr_multiplier = float(control.optimizer_action.lr_multiplier)
        lr_effective = self.base_lr * lr_multiplier

        for group in optimizer.param_groups:
            group["lr"] = lr_effective

        optimizer.zero_grad()
        loss = loss_closure()
        loss.backward()

        gradient = flatten_gradients(task_state.parameters)
        old_parameters = flatten_parameters(task_state.parameters)

        optimizer.step()

        update_vector = flatten_parameters(task_state.parameters) - old_parameters

        return UpdateResult(
            loss=loss.detach(),
            gradient=gradient,
            update_vector=update_vector,
            lr_effective=lr_effective,
        )