"""
Analytical optimization tasks for the first Hanno experiments.

These tasks are the recommended starting point for the restricted proof-of-
concept implementation because they isolate optimization behavior from the
extra complexity of datasets, minibatching, and deep model architectures.

This file implements a small family of differentiable objective functions:
- isotropic / anisotropic quadratic,
- Rosenbrock,
- saddle-type surface.

Each task:
- samples or constructs an initial parameter vector,
- exposes a differentiable scalar loss,
- and returns a TaskState that the environment can evolve over time.

The implementations here are intentionally clear and explicit rather than
highly abstract, because early interpretability matters more than cleverness.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hanno.core.types import TaskState
from hanno.tasks.base import BaseTask, TaskInfo


@dataclass
class AnalyticalTaskConfig:
    """
    Configuration shared by analytical objectives.

    Attributes
    ----------
    dimension:
        Number of optimizee parameters. Some objectives, such as the current
        Rosenbrock implementation, naturally use dimension 2.
    horizon:
        Number of environment steps per episode.
    init_scale:
        Scale used to sample the initial point.
    device:
        PyTorch device string.
    dtype:
        PyTorch dtype used for the parameters.
    """

    dimension: int
    horizon: int = 40
    init_scale: float = 1.5
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


class QuadraticTask(BaseTask):
    """
    Quadratic objective of the form:

        f(x) = 0.5 * x^T A x

    The matrix A is diagonal in this initial implementation, which keeps the
    objective easy to interpret. By choosing the diagonal entries, we can
    switch between isotropic and anisotropic curvature structures.
    """

    def __init__(
        self,
        diagonal: torch.Tensor,
        config: AnalyticalTaskConfig,
        name: str = "quadratic",
    ) -> None:
        super().__init__(name=name, horizon=config.horizon)
        self.diagonal = diagonal.to(device=config.device, dtype=config.dtype)
        self.config = config

    def reset(self, seed: int | None = None) -> TaskState:
        """
        Sample a fresh initial point for the quadratic objective.

        The initial point is drawn from a zero-mean Gaussian and scaled by
        init_scale. We wrap it in torch.nn.Parameter so optimizers can update
        it directly.
        """

        generator = torch.Generator(device=self.config.device)
        if seed is not None:
            generator.manual_seed(seed)

        x0 = self.config.init_scale * torch.randn(
            self.config.dimension,
            generator=generator,
            device=self.config.device,
            dtype=self.config.dtype,
        )

        return TaskState(
            parameters=torch.nn.Parameter(x0),
            step_index=0,
            metadata={
                "task_name": self.name,
                "dimension": self.config.dimension,
                "diagonal": self.diagonal.detach().cpu().tolist(),
            },
        )

    def compute_loss(self, task_state: TaskState) -> torch.Tensor:
        """
        Compute the quadratic loss.

        Because A is diagonal, x^T A x reduces to a weighted sum of x_i^2.
        """

        x = task_state.parameters
        return 0.5 * torch.sum(self.diagonal * x * x)

    def info(self) -> TaskInfo:
        """Return a compact summary of the quadratic task."""
        return TaskInfo(
            name=self.name,
            horizon=self.horizon,
            metadata={
                "dimension": self.config.dimension,
                "diagonal": self.diagonal.detach().cpu().tolist(),
            },
        )


class RosenbrockTask(BaseTask):
    """
    Two-dimensional Rosenbrock objective.

    The Rosenbrock function is a classic non-convex benchmark with a narrow
    curved valley. It is useful because successful optimization requires more
    than simply moving downhill greedily.
    """

    def __init__(
        self,
        config: AnalyticalTaskConfig,
        a: float = 1.0,
        b: float = 100.0,
        name: str = "rosenbrock",
    ) -> None:
        if config.dimension != 2:
            raise ValueError("RosenbrockTask currently requires dimension=2.")

        super().__init__(name=name, horizon=config.horizon)
        self.config = config
        self.a = a
        self.b = b

    def reset(self, seed: int | None = None) -> TaskState:
        """
        Sample a fresh initial point for the Rosenbrock function.

        Using a random start keeps the rollout setup consistent with the other
        analytical tasks, while still allowing reproducibility through a seed.
        """

        generator = torch.Generator(device=self.config.device)
        if seed is not None:
            generator.manual_seed(seed)

        x0 = self.config.init_scale * torch.randn(
            2,
            generator=generator,
            device=self.config.device,
            dtype=self.config.dtype,
        )

        return TaskState(
            parameters=torch.nn.Parameter(x0),
            step_index=0,
            metadata={
                "task_name": self.name,
                "a": self.a,
                "b": self.b,
            },
        )

    def compute_loss(self, task_state: TaskState) -> torch.Tensor:
        """
        Compute the Rosenbrock loss:

            f(x, y) = (a - x)^2 + b * (y - x^2)^2
        """

        x, y = task_state.parameters[0], task_state.parameters[1]
        return (self.a - x) ** 2 + self.b * (y - x ** 2) ** 2

    def info(self) -> TaskInfo:
        """Return a compact summary of the Rosenbrock task."""
        return TaskInfo(
            name=self.name,
            horizon=self.horizon,
            metadata={"a": self.a, "b": self.b},
        )


class SaddleTask(BaseTask):
    """
    Simple saddle-type objective in two dimensions.

    The chosen form is:

        f(x, y) = 0.5 * (x^2 - y^2)

    This surface is useful for studying how the controller behaves when the
    local geometry is not purely bowl-shaped.
    """

    def __init__(
        self,
        config: AnalyticalTaskConfig,
        name: str = "saddle",
    ) -> None:
        if config.dimension != 2:
            raise ValueError("SaddleTask currently requires dimension=2.")

        super().__init__(name=name, horizon=config.horizon)
        self.config = config

    def reset(self, seed: int | None = None) -> TaskState:
        """
        Sample a fresh initial point for the saddle objective.
        """

        generator = torch.Generator(device=self.config.device)
        if seed is not None:
            generator.manual_seed(seed)

        x0 = self.config.init_scale * torch.randn(
            2,
            generator=generator,
            device=self.config.device,
            dtype=self.config.dtype,
        )

        return TaskState(
            parameters=torch.nn.Parameter(x0),
            step_index=0,
            metadata={"task_name": self.name},
        )

    def compute_loss(self, task_state: TaskState) -> torch.Tensor:
        """
        Compute the saddle objective value.
        """

        x, y = task_state.parameters[0], task_state.parameters[1]
        return 0.5 * (x ** 2 - y ** 2)

    def info(self) -> TaskInfo:
        """Return a compact summary of the saddle task."""
        return TaskInfo(name=self.name, horizon=self.horizon, metadata={})


# ---------------------------------------------------------------------------
# Convenience factory helpers
# ---------------------------------------------------------------------------

def make_isotropic_quadratic(
    dimension: int = 2,
    curvature: float = 1.0,
    horizon: int = 40,
    init_scale: float = 1.5,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> QuadraticTask:
    """
    Build an isotropic quadratic task where all coordinates have equal curvature.
    """

    config = AnalyticalTaskConfig(
        dimension=dimension,
        horizon=horizon,
        init_scale=init_scale,
        device=device,
        dtype=dtype,
    )

    diagonal = curvature * torch.ones(dimension, dtype=dtype)
    return QuadraticTask(diagonal=diagonal, config=config, name="quadratic_isotropic")


def make_anisotropic_quadratic(
    diagonal_values: list[float] | tuple[float, ...],
    horizon: int = 40,
    init_scale: float = 1.5,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> QuadraticTask:
    """
    Build an anisotropic quadratic task with coordinate-dependent curvature.

    This is especially useful in early experiments because anisotropy exposes
    whether the controller behaves differently under ill-conditioned geometry.
    """

    diagonal = torch.tensor(diagonal_values, dtype=dtype)
    config = AnalyticalTaskConfig(
        dimension=len(diagonal_values),
        horizon=horizon,
        init_scale=init_scale,
        device=device,
        dtype=dtype,
    )
    return QuadraticTask(diagonal=diagonal, config=config, name="quadratic_anisotropic")


def make_rosenbrock(
    horizon: int = 60,
    init_scale: float = 1.2,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> RosenbrockTask:
    """
    Build the standard 2D Rosenbrock task.
    """

    config = AnalyticalTaskConfig(
        dimension=2,
        horizon=horizon,
        init_scale=init_scale,
        device=device,
        dtype=dtype,
    )
    return RosenbrockTask(config=config)


def make_saddle(
    horizon: int = 40,
    init_scale: float = 1.5,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SaddleTask:
    """
    Build the simple 2D saddle task.
    """

    config = AnalyticalTaskConfig(
        dimension=2,
        horizon=horizon,
        init_scale=init_scale,
        device=device,
        dtype=dtype,
    )
    return SaddleTask(config=config)
