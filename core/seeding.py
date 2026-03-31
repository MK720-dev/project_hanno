"""
Reproducibility helpers for Hanno experiments.

The first round of Hanno/HannoNet1 experiments needs to be easy to rerun and
compare. That means the project should centralize seed setting instead of
scattering it across scripts.

This file provides one small helper that seeds:
- Python's random module,
- NumPy,
- PyTorch CPU state,
- and PyTorch CUDA state when available.

The helper also optionally switches PyTorch into a more deterministic mode.
That mode can reduce nondeterminism, though it may slightly affect performance.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """
    Seed the main random number generators used by the project.

    Parameters
    ----------
    seed:
        The integer seed to apply across the supported libraries.
    deterministic:
        If True, PyTorch is asked to prefer deterministic behavior where
        possible. This is useful during debugging and early experiments.
    """

    # Seed Python's built-in random generator.
    random.seed(seed)

    # Seed NumPy's RNG for any numerical utilities or future data pipelines.
    np.random.seed(seed)

    # Seed PyTorch on CPU.
    torch.manual_seed(seed)

    # If CUDA is available, seed all visible CUDA devices as well.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Make Python's hash-based behaviors more stable across runs.
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # Ask PyTorch to use deterministic algorithms when possible.
        # This is particularly valuable in the early-stage prototype.
        torch.use_deterministic_algorithms(True, warn_only=True)

        # cuDNN-specific flags. These mainly matter when CUDA is used.
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
