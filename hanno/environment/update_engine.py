"""
Controlled optimizee update engine for Hanno / HannoNet1.

Role in the framework
---------------------
This file implements the inner-loop update mechanism that acts on the
optimizee parameters. This is intentionally separate from REINFORCE or any
other controller-training logic.

In HannoNet1:
- the controller observes optimizee-side diagnostics,
- the controller outputs optimizer-level control,
- the update engine applies that control to a fixed optimizer backbone,
- and the optimizee parameters are updated accordingly.

The current restricted proof-of-concept design only uses optimizer-level
learning-rate modulation. That means the controller does not replace SGD/Adam;
it only steers the effective step size through a multiplier.

Important conceptual boundary
-----------------------------
This file updates the optimizee.
It does NOT update the controller.
Controller updates belong to the outer RL training layer (e.g. REINFORCE).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hanno.core.types import ControlOutput, TaskState


@dataclass
class UpdateResult:
    """
    Result object returned after one controlled optimizee step.

    Keeping this structured helps the environment avoid recomputing useful
    quantities such as the gradient vector, the applied update, and the
    effective learning rate.
    """

    loss: torch.Tensor
    gradient: torch.Tensor
    update_vector: torch.Tensor
    lr_effective: float


class UpdateEngine:
    """
    Fixed-backbone optimizee update engine with controller-driven modulation.

    The backbone optimizer is conventional (SGD or Adam). The controller output
    is used to modulate the effective learning rate via a positive multiplier.
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
        """Create and attach a fresh backbone optimizer to the current optimizee."""

        params = [task_state.parameters]

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
        """Return the bound optimizer or raise an informative error."""

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

        The returned loss corresponds to the loss value before the update.
        The environment can then recompute the post-step loss explicitly.
        """

        optimizer = self._require_optimizer()

        # Read the optimizer-level control. This is the main restricted-scope
        # control channel in HannoNet1.
        lr_multiplier = float(control.optimizer_action.lr_multiplier)

        # Convert the controller multiplier into an effective learning rate.
        lr_effective = self.base_lr * lr_multiplier

        # Update every optimizer parameter group with the new effective LR.
        for group in optimizer.param_groups:
            group["lr"] = lr_effective

        optimizer.zero_grad()
        loss = loss_closure()
        loss.backward()

        # Clone the gradient now, before the optimizer mutates anything.
        gradient = task_state.parameters.grad.detach().clone().flatten()

        # Store the old parameters so we can recover the applied update vector.
        old_parameters = task_state.parameters.detach().clone()

        # Take one fixed-backbone optimizer step on the optimizee.
        optimizer.step()

        # Recover the applied parameter delta for diagnostics.
        update_vector = (task_state.parameters.detach() - old_parameters).flatten()

        return UpdateResult(
            loss=loss.detach(),
            gradient=gradient,
            update_vector=update_vector,
            lr_effective=lr_effective,
        )
