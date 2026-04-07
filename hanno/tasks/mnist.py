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


class MnistTask(BaseTask):
    """
    MNIST classification task with a freshly initialized small MLP per episode.
    """

    def __init__(
        self,
        config: MnistTaskConfig | None = None,
    ) -> None:
        self.config = config or MnistTaskConfig()
        super().__init__(name="mnist_mlp", horizon=self.config.horizon)

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

    def reset(self, seed: int | None = None) -> TaskState:
        """
        Reset the MNIST task for a fresh episode.

        This creates:
        - a new MLP optimizee,
        - a fresh episode-local DataLoader iterator,
        - and a cached first batch used for the initial loss/observation.
        """

        model = MnistMLP().to(self.config.device)
        loader = self._make_loader(seed=seed)
        batch_iter = iter(loader)

        task_state = TaskState(
            parameters=model,
            step_index=0,
            metadata={
                "task_name": self.name,
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
            },
        )
