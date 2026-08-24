"""Oscillator primitives shared by every synchrony model.

The thesis derives one phase mechanism (Chapter 3, eqs. kuramoto -> 
kuramoto-nd) but the code base had two: the VQA syncnet's scalar-angle
update and the coalitions gate's sphere update. Everything here is the
S^{d-1} form of that single mechanism, so a "phase" means the same thing
in `syncnet.py`, `phasebind.py` and `osc_tokens.py`:

    z_i  in S^{d-1}                                unit oscillator state
    <z_i, z_j>                                     alignment (= cos(theta_i - theta_j) at d=2)
    z_i <- Pi( z_i + dt * ( omega_i A z_i + tangent_i(sum_j K_ij v_j) ) )

where A is an antisymmetric generator (a rotation direction), tangent_i(v)
= v - <v, z_i> z_i is the projection onto the tangent space at z_i, and
Pi renormalises. At d = 2 the tangential component of z_j at z_i has
magnitude sin(theta_j - theta_i), so this is the Kuramoto coupling term
exactly, integrated with a projected Euler step.

Gate shapes:
    soft      (1 + <z_i, z_j>) / 2          the unsharpened thesis gate
    sharp     sigmoid(alpha <z_i,z_j> + b)  can actually close (rho.py)

Everything is a plain function over tensors so a model can compose the
step from whichever coupling terms it has (module-module, module-token,
token-token, stimulus) and still be running one dynamics.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# state

def random_unit(*shape: int, device=None, dtype=None) -> Tensor:
    """Uniform on S^{d-1}; the last dim is d."""
    z = torch.randn(*shape, device=device, dtype=dtype)
    return F.normalize(z, dim=-1)


def angle_to_unit(theta: Tensor) -> Tensor:
    """(..., ) angle -> (..., 2) unit vector."""
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)


def unit_to_angle(z: Tensor) -> Tensor:
    """(..., 2) unit vector -> (..., ) angle in [-pi, pi)."""
    return torch.atan2(z[..., 1], z[..., 0])


def tangent(z: Tensor, v: Tensor) -> Tensor:
    """Project v onto the tangent space of the sphere at z (both (..., d))."""
    return v - (v * z).sum(-1, keepdim=True) * z


def sphere_step(z: Tensor, velocity: Tensor, dt: float) -> Tensor:
    """Projected Euler step: Pi(z + dt * velocity)."""
    return F.normalize(z + dt * velocity, dim=-1)


def order_parameter(z: Tensor, dim: int = -2) -> Tensor:
    """||mean over `dim` of unit vectors||. z: (..., N, d) -> (...)."""
    return z.mean(dim=dim).norm(dim=-1)


# ---------------------------------------------------------------------------
# rotation generator (the natural-frequency term)

class SkewGenerator(nn.Module):
    """A learned antisymmetric d x d matrix A, so that omega * A z is a
    rotation at rate omega. At d = 2 this is fixed to the 90-degree
    rotation (A z = (-y, x)) so omega is an angular frequency exactly, as
    in the scalar model."""

    def __init__(self, d: int, learn: bool = True, init_scale: float = 0.1):
        super().__init__()
        self.d = d
        if d == 2:
            self.register_buffer('A', torch.tensor([[0.0, -1.0], [1.0, 0.0]]))
            self.raw = None
        else:
            raw = init_scale * torch.randn(d, d)
            if learn:
                self.raw = nn.Parameter(raw)
            else:
                self.register_buffer('raw', raw)
            self.A = None

    def matrix(self) -> Tensor:
        if self.d == 2:
            return self.A  # type: ignore[return-value]
        raw = self.raw
        A = raw - raw.t()
        # normalise so the largest rotation rate is 1 (omega scales it)
        return A / (A.norm() / math.sqrt(2) + 1e-6)

    def forward(self, z: Tensor, omega: Tensor) -> Tensor:
        """omega: broadcastable to z[..., 0]. Returns omega * A z (tangent)."""
        A = self.matrix()
        rot = torch.einsum('de,...e->...d', A, z)
        return omega.unsqueeze(-1) * rot


# ---------------------------------------------------------------------------
# gates

class GateShape(nn.Module):
    """Alignment -> channel strength in [0, 1].

    sharpen=False : (1 + a) / 2         (thesis eq. gate-phase; reaches 0
                                         only at antiphase)
    sharpen=True  : sigmoid(alpha a + b) (rho.py form; learnable alpha, b)
    """

    def __init__(self, sharpen: bool = False, alpha_init: float = 4.0,
                 bias_init: float = -1.0, learn: bool = True):
        super().__init__()
        self.sharpen = sharpen
        log_alpha = torch.tensor(float(math.log(alpha_init)))
        bias = torch.tensor(float(bias_init))
        if sharpen and learn:
            self.log_alpha = nn.Parameter(log_alpha)
            self.bias = nn.Parameter(bias)
        else:
            self.register_buffer('log_alpha', log_alpha)
            self.register_buffer('bias', bias)

    def forward(self, dots: Tensor) -> Tensor:
        if not self.sharpen:
            return 0.5 * (1.0 + dots)
        return torch.sigmoid(self.log_alpha.exp() * dots + self.bias)


def pairwise_dots(z: Tensor) -> Tensor:
    """(B, N, d) -> (B, N, N) alignment matrix <z_i, z_j>."""
    return torch.einsum('bid,bjd->bij', z, z)


def zero_diag(G: Tensor) -> Tensor:
    N = G.shape[-1]
    eye = torch.eye(N, device=G.device, dtype=G.dtype)
    return G * (1.0 - eye)


def straight_through_topk(G: Tensor, k: int, exclude_self: bool = True) -> Tensor:
    """Hard top-k over senders (last dim) with a straight-through gradient.
    Forward: one-hot mask of the k largest entries per receiver; backward:
    gradient of the soft gate. Used to *force* selective routing."""
    Gs = G
    if exclude_self:
        N = G.shape[-1]
        eye = torch.eye(N, device=G.device, dtype=torch.bool)
        Gs = G.masked_fill(eye, float('-inf'))
    idx = Gs.topk(k, dim=-1).indices
    hard = torch.zeros_like(G).scatter_(-1, idx, 1.0)
    return hard + (G - G.detach())


# ---------------------------------------------------------------------------
# coupling strengths between content states

class HebbianCoupling(nn.Module):
    """kappa_ij = tanh(<W h_i, W h_j> / sqrt(k)): signed, symmetric, no
    MLP. Like content attracts, unlike content repels -- the cheapest
    reading of 'coupling that depends on what the modules represent'."""

    def __init__(self, in_dim: int, key_dim: int = 32):
        super().__init__()
        self.proj = nn.Linear(in_dim, key_dim, bias=False)
        self.scale = 1.0 / math.sqrt(key_dim)

    def forward(self, h: Tensor) -> Tensor:            # (B, N, D) -> (B, N, N)
        k = self.proj(h)
        return torch.tanh(torch.einsum('bid,bjd->bij', k, k) * self.scale)


class MLPCoupling(nn.Module):
    """kappa_ij = tanh(MLP[h_i; h_j]) -- the thesis eq. phi form."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1), nn.Tanh())

    def forward(self, h: Tensor) -> Tensor:
        B, N, D = h.shape
        hi = h.unsqueeze(2).expand(B, N, N, D)
        hj = h.unsqueeze(1).expand(B, N, N, D)
        return self.net(torch.cat([hi, hj], -1)).squeeze(-1)


def grid_neighbours(S: int, eight: bool = False) -> Tensor:
    """(P, P) 0/1 adjacency of an S x S grid, row-major flattening."""
    ys, xs = torch.meshgrid(torch.arange(S), torch.arange(S), indexing='ij')
    ys, xs = ys.flatten(), xs.flatten()
    dy = (ys[:, None] - ys[None, :]).abs()
    dx = (xs[:, None] - xs[None, :]).abs()
    if eight:
        adj = (dy <= 1) & (dx <= 1)
    else:
        adj = (dy + dx) == 1
    adj = adj & ~torch.eye(S * S, dtype=torch.bool)
    return adj.float()
