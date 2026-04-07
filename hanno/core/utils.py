"""
Shared optimizee-side utility helpers for Hanno.

Role in the framework
---------------------
The original analytical-only prototype could get away with assuming that the
optimizee was a single torch.nn.Parameter vector. As soon as we move to MNIST
with an MLP, that assumption is no longer true: the optimizee becomes a full
neural network with many parameters.

This file centralizes the small helper functions needed to make the rest of the
framework agnostic to whether the optimizee is:
- a single Parameter (analytical phase), or
- a full nn.Module (MNIST and later CIFAR).

Keeping these helpers in one place avoids scattering "is this a Parameter or a
Module?" checks across the environment/update-engine code.
"""

from __future__ import annotations

import torch
from torch import nn


def optimizee_parameters(optimizee: nn.Module | nn.Parameter) -> list[nn.Parameter]:
    """
    Return the optimizee parameters as a concrete list.

    Parameters
    ----------
    optimizee:
        Either a single Parameter (analytical tasks) or an nn.Module
        (supervised-learning tasks).

    Returns
    -------
    list[nn.Parameter]
        Flat list of learnable parameters owned by the optimizee.
    """

    if isinstance(optimizee, nn.Parameter):
        return [optimizee]

    if isinstance(optimizee, nn.Module):
        return [p for p in optimizee.parameters() if p.requires_grad]

    raise TypeError(
        f"Unsupported optimizee type: {type(optimizee)!r}. "
        "Expected torch.nn.Parameter or torch.nn.Module."
    )


def get_optimizee_device(optimizee: nn.Module | nn.Parameter) -> torch.device:
    """
    Return the device on which the optimizee lives.
    """

    params = optimizee_parameters(optimizee)
    if not params:
        return torch.device("cpu")
    return params[0].device


def flatten_parameters(optimizee: nn.Module | nn.Parameter) -> torch.Tensor:
    """
    Flatten and concatenate the current optimizee parameters into one vector.

    This is useful for diagnostics such as parameter norms and applied update
    norms, where the environment wants a single compact numerical summary.
    """

    params = optimizee_parameters(optimizee)
    if not params:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat([p.detach().flatten() for p in params])


def flatten_gradients(optimizee: nn.Module | nn.Parameter) -> torch.Tensor:
    """
    Flatten and concatenate optimizee gradients into one vector.

    Missing gradients are replaced with zeros of matching shape so the returned
    vector always has the same overall parameter layout.
    """

    params = optimizee_parameters(optimizee)
    if not params:
        return torch.zeros(0, dtype=torch.float32)

    pieces = []
    for p in params:
        if p.grad is None:
            pieces.append(torch.zeros_like(p.detach()).flatten())
        else:
            pieces.append(p.grad.detach().flatten())
    return torch.cat(pieces)