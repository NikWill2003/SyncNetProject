"""The operators of the intervention suites, implemented once as pure
functions so every composition is intervened on identically.

    phase_shuffle    permute the module phases       is routing by address?
    (freeze / zero live in the model loop: freeze skips the Phi step,
     zero replaces z with a constant row)
    anchor_shuffle   permute the identity priors     is routing by name?
"""

from __future__ import annotations

import torch
from torch import Tensor


def phase_shuffle(z: Tensor, n_modules: int) -> Tensor:
    """Permute the slot rows' phases within each sample; the head keeps its own."""
    perm = torch.randperm(n_modules, device=z.device)
    return torch.cat([z[:, perm], z[:, n_modules:]], 1)


def anchor_shuffle(n_modules: int, device) -> Tensor:
    """A permutation for the per-module anchor priors (identity model only)."""
    return torch.randperm(n_modules, device=device)
