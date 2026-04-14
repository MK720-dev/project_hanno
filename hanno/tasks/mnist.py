"""
MNIST task wrapper for Hanno / HannoNet1.

Role in the framework
---------------------
This file turns MNIST classification into a task object compatible with the
existing Hanno environment abstraction.

A single Hanno episode consists of a fixed number of minibatch updates on a
freshly initialized small MLP. At each step:
- the current cached minibatch is used to compute the optimizee loss,
- the controller chooses a control action,
- the update engine updates the MLP,
- reward and observation are computed,
- then the task advances to the next minibatch for the next step.

This preserves the same inner/outer loop structure used in the analytical
phase while introducing:
- a real dataset,
- stochastic minibatches,
- and a standard supervised-learning loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from hanno.core.types import TaskState
from hanno.tasks.base import BaseTask, TaskInfo
from hanno.tasks.models.mlp import MnistMLP


class FlexibleMnistMLP(nn.Module):
    """
    Simple configurable MLP used to broaden the MNIST architecture family
    without changing the rest of the Hanno stack.
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...],
        input_dim: int = 28 * 28,
        output_dim: int = 10,
    ) -> None:
        super().__init__()

        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one hidden layer size.")

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.network(x)


MNIST_MLP_VARIANTS: dict[str, tuple[int, ...] | None] = {
    # Baseline variant keeps using the existing model class so the original
    # proof-of-concept path remains available unchanged.
    "small": None,
    # Additional variants used for broader transfer experiments.
    "narrow": (128,),
    "wide": (512, 256),
    "deep": (256, 128, 64),
    "bottleneck": (256, 64),
    "very_wide": (768, 384),
}


@dataclass
class MnistTaskConfig:
    """
    Configuration for the MNIST task.
    """

    horizon: int = 50
    batch_size: int = 64
    train: bool = True
    data_root: str = "./data"
    shuffle: bool = True
    device: str = "cpu"
    model_variant: str = "small"


class MnistTask(BaseTask):
    """
    MNIST classification task with a freshly initialized small MLP per episode.
    """

    def __init__(
        self,
        config: MnistTaskConfig | None = None,
    ) -> None:
        self.config = config or MnistTaskConfig()

        if self.config.model_variant not in MNIST_MLP_VARIANTS:
            valid_variants = ", ".join(sorted(MNIST_MLP_VARIANTS))
            raise ValueError(
                f"Unknown MNIST model_variant '{self.config.model_variant}'. "
                f"Expected one of: {valid_variants}"
            )

        task_name = f"mnist_mlp_{self.config.model_variant}"
        super().__init__(name=task_name, horizon=self.config.horizon)

        self.loss_fn = nn.CrossEntropyLoss()

        transform = transforms.ToTensor()
        self.dataset = datasets.MNIST(
            root=self.config.data_root,
            train=self.config.train,
            download=True,
            transform=transform,
        )

    def _make_loader(self, seed: int | None) -> DataLoader:
        """
        Build a fresh DataLoader for one episode.

        We create a fresh loader so that episode resets are well-defined and
        reproducible under a given seed.
        """

        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=0,
            generator=generator,
        )

    def _next_batch(self, task_state: TaskState) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the next minibatch for the current episode, cycling the iterator if
        necessary.
        """

        batch_iter = task_state.metadata["batch_iter"]

        try:
            x, y = next(batch_iter)
        except StopIteration:
            loader = self._make_loader(seed=None)
            batch_iter = iter(loader)
            task_state.metadata["batch_iter"] = batch_iter
            x, y = next(batch_iter)

        x = x.to(self.config.device)
        y = y.to(self.config.device)
        return x, y

    def _make_model(self) -> nn.Module:
        """
        Build the fresh episode-local optimizee for the requested architecture
        variant.
        """

        hidden_dims = MNIST_MLP_VARIANTS[self.config.model_variant]
        if hidden_dims is None:
            return MnistMLP()
        return FlexibleMnistMLP(hidden_dims=hidden_dims)

    def reset(self, seed: int | None = None) -> TaskState:
        """
        Reset the MNIST task for a fresh episode.

        This creates:
        - a new MLP optimizee,
        - a fresh episode-local DataLoader iterator,
        - and a cached first batch used for the initial loss/observation.
        """

        model = self._make_model().to(self.config.device)
        loader = self._make_loader(seed=seed)
        batch_iter = iter(loader)

        task_state = TaskState(
            parameters=model,
            step_index=0,
            metadata={
                "task_name": self.name,
                "model_variant": self.config.model_variant,
                "model_class": model.__class__.__name__,
                "batch_iter": batch_iter,
            },
        )

        # Cache the first batch immediately so reset() can compute the initial
        # loss before any controller action is taken.
        x, y = self._next_batch(task_state)
        task_state.metadata["current_batch"] = (x, y)

        return task_state

    def compute_loss(self, task_state: TaskState) -> torch.Tensor:
        """
        Compute the current MNIST cross-entropy loss on the cached minibatch.

        The same cached batch is used both for the pre-update loss and the
        post-update loss within a single environment step. Only after the step
        is complete does the task advance to the next batch.
        """

        model = task_state.parameters
        if not isinstance(model, nn.Module):
            raise TypeError("MnistTask expects TaskState.parameters to be an nn.Module.")

        x, y = task_state.metadata["current_batch"]
        logits = model(x)
        return self.loss_fn(logits, y)

    def advance(self, task_state: TaskState) -> None:
        """
        Advance to the next cached minibatch after a completed step.
        """

        x, y = self._next_batch(task_state)
        task_state.metadata["current_batch"] = (x, y)

    def info(self) -> TaskInfo:
        """
        Return a compact summary of the MNIST task.
        """

        return TaskInfo(
            name=self.name,
            horizon=self.horizon,
            metadata={
                "batch_size": self.config.batch_size,
                "train_split": self.config.train,
                "model_variant": self.config.model_variant,
            },
        )
