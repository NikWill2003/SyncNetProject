"""The oscillator field: binding-by-synchrony's physical layer.

Every cell of the conv feature map carries FIELD_GROUPS unit vectors on
S^{FIELD_OSC_D - 1}. T_FIELD Euler steps of natural rotation + learned local
coupling (a 5x5 convolution through which phase propagates) + feature
stimulus, tangent-projected and renormalised, leave cells of one object
rotating together: the relation "belongs together" gets its own variable,
written by dynamics, erased next image.

Known limit, stated where it lives: on stimulus-starved inputs (sparse thin
glyphs at low contrast) the field enters its globally synchronised trivial
fixed point and every downstream stage inherits a constant. The SQOOP
negative is the documented study of exactly that; the fixed-point detector
in analysis/sync_metrics reads the signature.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

FIELD_CH = 64
FIELD_HIDDEN = 48
FIELD_GROUPS = 16
FIELD_OSC_D = 4
T_FIELD = 8
DT_FIELD = 1.0


class FieldEncoder(nn.Module):
    """Two stride-2 stages then a 3x3 head: img -> (B, FIELD_CH, S, S)."""

    def __init__(self, img_size: int, hidden: int = FIELD_HIDDEN,
                 n_down: int = 2, out_ch: int = FIELD_CH):
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
    """FIELD_GROUPS unit FIELD_OSC_D-vectors per position, evolved T steps:
    natural rotation + conv coupling + feature stimulus, tangent-projected,
    renormalised. Phase init comes from the features (z_head)."""

    def __init__(self, fch: int = FIELD_CH, osc_dim: int = FIELD_OSC_D,
                 n_groups: int = FIELD_GROUPS, T: int = T_FIELD,
                 dt: float = DT_FIELD, ksize: int = 5):
        super().__init__()
        self.d, self.K, self.C = osc_dim, n_groups, osc_dim * n_groups
        self.T, self.dt = T, dt
        self.z_head = nn.Conv2d(fch, self.C, 1)
        self.stim = nn.Conv2d(fch, self.C, 1)
        self.J = nn.Conv2d(self.C, self.C, ksize, padding=ksize // 2, bias=False)
        nn.init.normal_(self.J.weight, std=0.02)
        self.omega_raw = nn.Parameter(torch.randn(n_groups, osc_dim, osc_dim) * 0.1)
        self.omega_scale = 0.1

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
        B, C, S, _ = z.shape
        return z.view(B, self.K, self.d, S * S).permute(0, 3, 1, 2)          # (B, P, K, d)

    def forward(self, f: Tensor) -> Tensor:
        z = self.normalise(self.z_head(f))
        c = self.stim(f)
        for _ in range(self.T):
            drive = self.J(z) + c
            z = self.normalise(z + self.dt * (self.rotate(z) + self.tangent(z, drive)))
        return z


class AdaptedTrunk(nn.Module):
    """A common encoder adapted to the field interface: 1x1 conv to
    FIELD_CH so the oscillator machinery is unchanged. Grid size follows
    the base encoder (the field is size-agnostic; pos_emb reads .spatial)."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.adapt = nn.Conv2d(base.ch, FIELD_CH, 1)
        self.spatial = base.spatial

    def forward(self, x: Tensor) -> Tensor:
        return self.adapt(self.base(x))


def build_field_trunk(enc_cfg, img_size: int, dataset: str) -> nn.Module:
    """Default (enc_cfg absent or name 'field'): the FieldEncoder, exactly
    as always -- the thesis cells never change. A named common encoder
    routes through build_image_encoder + AdaptedTrunk."""
    name = None
    if enc_cfg:
        name = enc_cfg.get('name') if isinstance(enc_cfg, dict) else getattr(enc_cfg, 'name', None)
    if name in (None, 'field'):
        return FieldEncoder(img_size)
    from ..common.img_enc import build_image_encoder
    return AdaptedTrunk(build_image_encoder(dict(enc_cfg), dataset))
