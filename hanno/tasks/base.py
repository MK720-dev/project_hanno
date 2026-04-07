"""
Base task abstractions for Hanno.

A task is the optimizee side of the system. It answers questions such as:
- what parameters are being optimized,
- how are they initialized,
- how is the loss computed,
- and how long should the episode last?

The first implementation uses simple analytical objectives, but the same
contract is designed to later support neural networks trained on datasets.

The main abstraction here is intentionally small:
- reset the task,
- compute a loss from the current parameters,
- expose a horizon,
- and provide task metadata for logging/debugging.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from hanno.core.types import TaskState


@dataclass
class TaskInfo:
    """
    Lightweight metadata about a task instance.

    This is useful for experiment logs, debugging, and future task registries.
    """

    name: str
    horizon: int
    metadata: dict[str, Any]


class BaseTask(ABC):
    """
    Abstract base class for all optimizee tasks.

    Each concrete task must define how parameters are initialized and how the
    scalar objective/loss is computed from those parameters.
    """

    def __init__(self, name: str, horizon: int) -> None:
        self._name = name
        self._horizon = horizon

    @property
    def name(self) -> str:
        """Return the human-readable task name."""
        return self._name

    @property
    def horizon(self) -> int:
        """Return the fixed episode horizon for this task."""
        return self._horizon

    @abstractmethod
    def reset(self, seed: int | None = None) -> TaskState:
        """
        Create and return a fresh task state for a new episode.

        A reset should initialize the optimizee parameters and attach any
        relevant metadata describing the sampled task instance.
        """

    @abstractmethod
    def compute_loss(self, task_state: TaskState) -> torch.Tensor:
        """
        Compute the scalar objective from the current task state.

        The returned tensor should be differentiable with respect to the task
        parameters so the update engine can backpropagate through it.
        """

    def info(self) -> TaskInfo:
        """
        Return a compact summary of the task.

        Subclasses can override this if they want to expose richer metadata.
        """

        return TaskInfo(name=self.name, horizon=self.horizon, metadata={})
