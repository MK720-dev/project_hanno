"""
Small MLP optimizee for Hanno's MNIST phase.

Role in the framework
---------------------
This is the first neural-network optimizee used after the analytical phase.
It is intentionally small and standard so that the focus remains on the
controller's optimization behavior rather than architectural novelty.

Architecture
------------
Input:  28 x 28 grayscale image flattened to 784
Hidden: 128 -> ReLU -> 64 -> ReLU
Output: 10 logits for MNIST digit classes
"""

from __future__ import annotations

import torch
from torch import nn


class MnistMLP(nn.Module):
    """
    Small MLP for MNIST classification.
    """

    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        output_dim: int = 10,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        MNIST images arrive as [batch, 1, 28, 28], so we flatten them to
        [batch, 784] before passing them through the dense network.
        """

        x = x.view(x.size(0), -1)
        return self.network(x)
