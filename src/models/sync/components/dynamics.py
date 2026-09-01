"""Phi -- where addresses come from.

One Kuramoto step per model step: shared-axis natural rotation at per-row
rates, signed content coupling kappa_ij = tanh(MLP([h_i; h_j])), and the
STIMULUS W_s h_i -- the term that lets a module place its own phase from
what it represents. Computed-vs-discovered addresses is the single most
consequential design choice in the canonical model; the static and open
address modes that ablate it live at the model level (they are address
sources, not dynamics).

The scalar circle of the gated model is d = 2 of the same mathematics,
not a separate implementation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def tangent(z: Tensor, v: Tensor) -> Tensor:
    """Project v onto the tangent space of the unit sphere at z."""
    return v - (v * z).sum(-1, keepdim=True) * z


class SkewGenerator(nn.Module):
    """A learned antisymmetric generator: omega * A z rotates z at rate
    omega. A is normalised so its largest rotation rate is 1, making omega
    an angular frequency."""

    def __init__(self, d: int, init_scale: float = 0.1):
        super().__init__()
        self.raw = nn.Parameter(init_scale * torch.randn(d, d))

    def forward(self, z: Tensor, omega: Tensor) -> Tensor:
        A = self.raw - self.raw.t()
        A = A / (A.norm() / math.sqrt(2) + 1e-6)
        return omega.unsqueeze(-1) * torch.einsum('de,bne->bnd', A, z)


class KuramotoStep(nn.Module):
    """z <- Pi( z + dt ( omega A z + P_z( (K * kappa(h)) z + W_s h ) ) )."""

    def __init__(self, n_rows: int, module_dim: int, phase_dim: int,
                 dt: float, k_hidden: int = 64, stimulus: bool = True):
        super().__init__()
        self.dt, self.stimulus = dt, stimulus
        self.omega = nn.Parameter(torch.zeros(n_rows))
        self.K = nn.Parameter(torch.ones(n_rows, n_rows))
        self.k_mlp = nn.Sequential(nn.Linear(2 * module_dim, k_hidden), nn.GELU(),
                                   nn.Linear(k_hidden, 1), nn.Tanh())
        self.stim = nn.Linear(module_dim, phase_dim)
        self.gen = SkewGenerator(phase_dim)

    def forward(self, z: Tensor, h: Tensor) -> Tensor:
        B, N, dm = h.shape
        vel = self.gen(z, self.omega.to(z.dtype).unsqueeze(0).expand(B, N))
        hi = h.unsqueeze(2).expand(B, N, N, dm)
        hj = h.unsqueeze(1).expand(B, N, N, dm)
        kap = self.k_mlp(torch.cat([hi, hj], -1)).squeeze(-1)                # signed, in (-1, 1)
        pull = torch.einsum('bij,bjd->bid', self.K.to(z.dtype).unsqueeze(0) * kap, z)
        vel = vel + tangent(z, pull)
        if self.stimulus:
            vel = vel + tangent(z, self.stim(h))                             # the stimulus drive
        return F.normalize(z + self.dt * vel, dim=-1)
