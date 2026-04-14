"""
CIFAR-10 task wrapper for Hanno / HannoNet1.

Role in the framework
---------------------
This file turns CIFAR-10 classification into a task object compatible with the
existing Hanno environment abstraction.

A single Hanno episode consists of a fixed number of minibatch updates on a
freshly initialized small CNN. At each step:
- the current cached minibatch is used to compute the optimizee loss,
- the controller chooses a control action,
- the update engine updates the CNN,
- reward and observation are computed,
- then the task advances to the next minibatch for the next step.

This preserves the same inner/outer loop structure used in the MNIST stage
while moving to a standard convolutional benchmark with spatial structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from hanno.core.types import TaskState
from hanno.tasks.base import BaseTask, TaskInfo


class FlexibleCifarCNN(nn.Module):
    """
    Simple configurable CNN used to define a small CIFAR-10 architecture family
    without changing the rest of the Hanno stack.
    """

    def __init__(
        self,
        conv_channels: tuple[int, ...],
        fc_hidden_dim: int,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        if len(conv_channels) < 2:
            raise ValueError("conv_channels must contain at least two entries.")

        conv_layers: list[nn.Module] = []
        in_channels = 3

        for block_idx, out_channels in enumerate(conv_channels):
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            if block_idx < len(conv_channels) - 1:
                conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.features = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        flattened_dim = conv_channels[-1] * 4 * 4

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


CIFAR10_CNN_VARIANTS: dict[str, tuple[tuple[int, ...], int]] = {
    "small": ((32, 64), 128),
    "wide": ((48, 96), 192),
    "deep": ((32, 64, 96), 160),
    "narrow": ((24, 48), 96),
    "bottleneck": ((32, 48, 64), 96),
    "very_wide": ((64, 128), 256),
}


@dataclass
class Cifar10TaskConfig:
    """
    Configuration for the CIFAR-10 task.
    """

    horizon: int = 75
    batch_size: int = 64
    train: bool = True
    data_root: str = "./data"
    shuffle: bool = True
    device: str = "cpu"
    model_variant: str = "small"


class Cifar10Task(BaseTask):
    """
    CIFAR-10 classification task with a freshly initialized small CNN per
    episode.
    """

    def __init__(
        self,
        config: Cifar10TaskConfig | None = None,
    ) -> None:
        self.config = config or Cifar10TaskConfig()

        if self.config.model_variant not in CIFAR10_CNN_VARIANTS:
            valid_variants = ", ".join(sorted(CIFAR10_CNN_VARIANTS))
            raise ValueError(
                f"Unknown CIFAR-10 model_variant '{self.config.model_variant}'. "
                f"Expected one of: {valid_variants}"
            )

        task_name = f"cifar10_cnn_{self.config.model_variant}"
        super().__init__(name=task_name, horizon=self.config.horizon)

        self.loss_fn = nn.CrossEntropyLoss()

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
            ]
        )
        self.dataset = datasets.CIFAR10(
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

        conv_channels, fc_hidden_dim = CIFAR10_CNN_VARIANTS[self.config.model_variant]
        return FlexibleCifarCNN(
            conv_channels=conv_channels,
            fc_hidden_dim=fc_hidden_dim,
        )

    def reset(self, seed: int | None = None) -> TaskState:
        """
        Reset the CIFAR-10 task for a fresh episode.

        This creates:
        - a new CNN optimizee,
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
        Compute the current CIFAR-10 cross-entropy loss on the cached minibatch.

        The same cached batch is used both for the pre-update loss and the
        post-update loss within a single environment step. Only after the step
        is complete does the task advance to the next batch.
        """

        model = task_state.parameters
        if not isinstance(model, nn.Module):
            raise TypeError("Cifar10Task expects TaskState.parameters to be an nn.Module.")

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
        Return a compact summary of the CIFAR-10 task.
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
