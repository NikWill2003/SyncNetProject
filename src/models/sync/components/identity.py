"""I -- who the modules are, answered in one place: the recurrent cells
(one shared, or M private), the identity embeddings, and whether the anchor
prior is shared or per-module.

The invariant that keeps privatisation honest: the identity component never
touches the medium. Message projection, stimulus map and coupling MLP are
model-level and shared -- private minds, one radio standard -- so identity
cannot leak into the protocol even when the modules are individuals.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class Exchangeable(nn.Module):
    """The canonical identity: shared cell, shared anchor prior, no
    embeddings. With interchangeable modules, routing provably cannot be
    identity-based -- the attribution guarantee behind the shuffle
    intervention."""

    per_module_anchors = False

    def __init__(self, n_rows: int, in_dim: int, module_dim: int, n_modules: int):
        super().__init__()
        self.cell = nn.GRUCell(in_dim, module_dim)
        self.embeds = None

    def step(self, inp: Tensor, h: Tensor) -> Tensor:
        B, N, dm = h.shape
        return self.cell(inp.reshape(B * N, -1), h.reshape(B * N, dm)).reshape(B, N, dm)

    def embed(self, h_slots: Tensor) -> Tensor:
        return h_slots


class PhaseNative(nn.Module):
    """Identity in phase space: per-module anchor priors give private
    parameters a stable referent (module k habitually wins the
    neighbourhoods its prior prefers), private GRU cells give the referent
    somewhere to matter, and embeddings mark the rows. The two factors are
    independently constructable because the interaction claim needs the
    single-factor cells."""

    def __init__(self, n_rows: int, in_dim: int, module_dim: int, n_modules: int,
                 private_cells: bool = True, per_module_anchors: bool = True):
        super().__init__()
        self.per_module_anchors = per_module_anchors
        self.private = private_cells
        if private_cells:
            self.cells = nn.ModuleList(nn.GRUCell(in_dim, module_dim) for _ in range(n_rows))
        else:
            self.cell = nn.GRUCell(in_dim, module_dim)
        self.embeds = nn.Parameter(torch.randn(n_modules, module_dim) / module_dim ** 0.5)

    def step(self, inp: Tensor, h: Tensor) -> Tensor:
        B, N, dm = h.shape
        if self.private:
            return torch.stack([self.cells[k](inp[:, k], h[:, k]) for k in range(N)], 1)
        return self.cell(inp.reshape(B * N, -1), h.reshape(B * N, dm)).reshape(B, N, dm)

    def embed(self, h_slots: Tensor) -> Tensor:
        return h_slots + self.embeds.unsqueeze(0)
