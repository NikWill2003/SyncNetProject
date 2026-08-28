"""An oscillator field over a conv feature map, factored out of OscField
so other models can run the same dynamics. The maths is OscField's:

    z <- Pi( z + dt [ Omega z + Proj_z( J * z + c ) ] )

with K oscillators of dimension d per position, a learned convolutional
coupling J, a per-group antisymmetric generator Omega, and a conditional
stimulus c computed from the (question-modulated) features.

`osc_field.py` is deliberately left untouched (runs in flight import it);
this file duplicates the ~40 lines rather than refactoring them.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FieldEncoder(nn.Module):
    """n_down stride-2 stages then a 3x3 conv to out_ch."""

    def __init__(self, img_size: int, hidden: int, n_down: int, out_ch: int):
        super().__init__()
        layers: list[nn.Module] = []
        cin, s = 3, img_size
        for _ in range(n_down):
            layers += [nn.Conv2d(cin, hidden, 3, stride=2, padding=1),
                       nn.GroupNorm(8, hidden), nn.SiLU()]
            cin, s = hidden, (s + 1) // 2
        layers += [nn.Conv2d(cin, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.SiLU(),
                   nn.Conv2d(hidden, out_ch, 3, padding=1)]
        self.net = nn.Sequential(*layers)
        self.spatial = s

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class OscillatorField(nn.Module):
    """Field dynamics only: takes features f (B, fch, S, S), returns the
    final oscillator state z (B, K*d, S, S) and optionally the trajectory."""

    def __init__(self, fch: int, osc_dim: int = 4, n_groups: int = 16, T: int = 8,
                 dt: float = 1.0, ksize: int = 5, coupling: str = 'conv',
                 stimulus: bool = True, learn_omega: bool = True,
                 omega_scale: float = 0.1, z_init: str = 'feature', step_max_deg: float = 0.0) -> None:
        super().__init__()
        self.d, self.K, self.C = osc_dim, n_groups, osc_dim * n_groups
        self.T, self.dt = T, dt
        self.step_max_deg = step_max_deg
        self.coupling, self.stimulus, self.z_init = coupling, stimulus, z_init
        self.omega_scale = omega_scale
        self.z_head = nn.Conv2d(fch, self.C, 1)
        if stimulus:
            self.stim = nn.Conv2d(fch, self.C, 1)
        if coupling == 'conv':
            self.J = nn.Conv2d(self.C, self.C, ksize, padding=ksize // 2, bias=False)
            nn.init.normal_(self.J.weight, std=0.02)
        elif coupling != 'none':
            raise ValueError(f'unknown coupling {coupling!r}')
        raw = torch.randn(n_groups, osc_dim, osc_dim) * 0.1
        if learn_omega:
            self.omega_raw = nn.Parameter(raw)
        else:
            self.register_buffer('omega_raw', raw)

    # per-group helpers ----------------------------------------------
    def normalise(self, z: Tensor) -> Tensor:
        B, C, S, _ = z.shape
        return F.normalize(z.view(B, self.K, self.d, S, S), dim=2).view(B, C, S, S)

    def tangent(self, z: Tensor, v: Tensor) -> Tensor:
        B, C, S, _ = z.shape
        zg, vg = z.view(B, self.K, self.d, S, S), v.view(B, self.K, self.d, S, S)
        return (vg - (vg * zg).sum(2, keepdim=True) * zg).reshape(B, C, S, S)

    def rotate(self, z: Tensor) -> Tensor:
        B, C, S, _ = z.shape
        A = (self.omega_raw - self.omega_raw.transpose(-1, -2)) * self.omega_scale
        return torch.einsum('kde,bkeij->bkdij', A, z.view(B, self.K, self.d, S, S)).reshape(B, C, S, S)

    def to_tokens(self, z: Tensor) -> Tensor:
        """(B, C, S, S) -> (B, P, K, d)"""
        B, C, S, _ = z.shape
        return z.view(B, self.K, self.d, S * S).permute(0, 3, 1, 2)

    # ------------------------------------------------------------------
    def forward(self, f: Tensor, T: int | None = None, freeze: bool = False,
                shuffle: bool = False, return_trace: bool = False):
        B, _, S, _ = f.shape
        T = self.T if T is None else T
        if self.z_init == 'feature':
            z = self.normalise(self.z_head(f))
        else:
            z = self.normalise(torch.randn(B, self.C, S, S, device=f.device, dtype=f.dtype))
        if shuffle:
            perm = torch.argsort(torch.rand(B, S * S, device=f.device), dim=1)
            z = z.flatten(2).gather(2, perm.unsqueeze(1).expand(-1, self.C, -1)).view(B, self.C, S, S)
        c = self.stim(f) if self.stimulus else None
        trace = []
        if not freeze:
            for _ in range(T):
                drive = torch.zeros_like(z)
                if self.coupling == 'conv':
                    drive = drive + self.J(z)
                if c is not None:
                    drive = drive + c
                vel = self.rotate(z) + self.tangent(z, drive)
                if self.step_max_deg > 0:
                    Bz, Cz, Sz, _ = z.shape
                    step = (self.dt * vel).view(Bz, self.K, self.d, Sz, Sz)
                    cap = math.tan(math.radians(self.step_max_deg))
                    n = step.norm(dim=2, keepdim=True)
                    vel = (step * torch.clamp(cap / (n + 1e-8), max=1.0)).view(Bz, Cz, Sz, Sz) / self.dt
                z = self.normalise(z + self.dt * vel)
                if return_trace:
                    trace.append(z.detach())
        return (z, trace) if return_trace else z
