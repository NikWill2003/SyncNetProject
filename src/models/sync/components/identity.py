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
        # private_cells: True -> one GRU per row (the original private minds)
        #               False -> one shared GRU
        #           'residual' -> shared GRU + zero-initialised per-row deltas.
        # The residual form is functionally identical to the shared cell at
        # init, so assembly odds start at the shared-cell rate, but the shared
        # weights keep receiving every row's gradient (the pooling that plain
        # private cells give up, and the suspected cause of their collapse).
        # ||delta_k|| over training measures when slots become individuals.
        self.private = private_cells is True
        self.residual = private_cells == 'residual'
        if self.private:
            self.cells = nn.ModuleList(nn.GRUCell(in_dim, module_dim) for _ in range(n_rows))
        else:
            self.cell = nn.GRUCell(in_dim, module_dim)
        if self.residual:
            self.d_ih = nn.Parameter(torch.zeros(n_rows, 3 * module_dim, in_dim))
            self.d_hh = nn.Parameter(torch.zeros(n_rows, 3 * module_dim, module_dim))
            self.d_bias = nn.Parameter(torch.zeros(n_rows, 3 * module_dim))
        self.embeds = nn.Parameter(torch.randn(n_modules, module_dim) / module_dim ** 0.5)

    def step(self, inp: Tensor, h: Tensor) -> Tensor:
        B, N, dm = h.shape
        if self.private:
            return torch.stack([self.cells[k](inp[:, k], h[:, k]) for k in range(N)], 1)
        if self.residual:
            return self._residual_step(inp, h)
        return self.cell(inp.reshape(B * N, -1), h.reshape(B * N, dm)).reshape(B, N, dm)

    def _residual_step(self, inp: Tensor, h: Tensor) -> Tensor:
        """GRU with per-row weight deltas: W_k = W_shared + delta_k."""
        w_ih = self.cell.weight_ih.unsqueeze(0) + self.d_ih                  # (N, 3dm, in)
        w_hh = self.cell.weight_hh.unsqueeze(0) + self.d_hh
        b = self.cell.bias_ih.unsqueeze(0) + self.cell.bias_hh.unsqueeze(0) + self.d_bias
        gi = torch.einsum('bnf,nof->bno', inp, w_ih)
        gh = torch.einsum('bnd,nod->bno', h, w_hh)
        gi_r, gi_z, gi_n = gi.chunk(3, dim=-1)
        gh_r, gh_z, gh_n = gh.chunk(3, dim=-1)
        b_r, b_z, b_n = b.chunk(3, dim=-1)
        r = torch.sigmoid(gi_r + gh_r + b_r)
        z = torch.sigmoid(gi_z + gh_z + b_z)
        n = torch.tanh(gi_n + r * gh_n + b_n)
        return (1 - z) * n + z * h

    def delta_norm(self) -> float:
        """Mean ||delta_k|| -- how far slots have individuated (residual only)."""
        if not self.residual:
            return 0.0
        return float(self.d_ih.flatten(1).norm(dim=1).mean()
                     + self.d_hh.flatten(1).norm(dim=1).mean())

    def embed(self, h_slots: Tensor) -> Tensor:
        return h_slots + self.embeds.unsqueeze(0)
