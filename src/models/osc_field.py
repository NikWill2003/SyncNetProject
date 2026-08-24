"""OscField: synchrony at the level of the feature map, with no modules.

An oscillator field in the AKOrN form: every position of a conv feature
map carries K oscillators of dimension d, coupled through a learned
convolution J and driven by a conditional stimulus c computed from the
(question-modulated) features,

    z <- Pi( z + dt [ Omega z + Proj_z( J * z + c ) ] ),

run for T steps so that phases group into objects (or into whatever the
question makes relevant). Nothing is tokenised into modules and nothing
is gated; the question is whether token-level synchrony alone supports
relational reasoning on this task, i.e. whether the module layer in the
syncnet is earning its place.

Readout is where the phase is used ('sync'):
  hop 1  each of S slots finds its object by content attention and
         takes the attended phase as an anchor  phi_s
  hop 2  the slot then reads every position in phase with its anchor,
         alpha_sp propto exp( lambda <phi_s, z_p> + beta' cos(q'_s, X_p) )
so "read the coalition of the queried object" is the operation, and the
anchor alignments <phi_s, phi_s'> between slots are also fed to the head.
'content' readout drops hop 2 and the anchors: same model, no phase used.

Runtime overrides: t_override, phase_override {freeze, shuffle, lambda0}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.config import ModelConfig
from .question_encoders import QuestionEncoder
from .oscillators import sphere_step, tangent


@dataclass
class OscFieldConfig(ModelConfig):
    q_conditioning: str = 'film'        # film | broadcast_cat | token
    q_emb_dim: int = 32
    # field encoder: n_down stride-2 stages, then a 3x3 conv to field_ch
    hidden: int = 48
    n_down: int = 2                     # 75px: 1 -> 38x38, 2 -> 19x19, 3 -> 10x10
    field_ch: int = 64
    use_pos_emb: bool = True
    # oscillators: n_groups oscillators of dim osc_dim per position
    osc_dim: int = 4
    n_groups: int = 16
    T: int = 8
    dt: float = 1.0
    ksize: int = 5                      # coupling kernel
    coupling: str = 'conv'              # conv | none
    stimulus: bool = True
    learn_omega: bool = True
    omega_scale: float = 0.1
    z_init: str = 'feature'             # feature | random
    # readout
    readout: str = 'sync'               # sync | content
    n_slots: int = 2
    beta_init: float = 5.0
    lambda_init: float = 4.0
    hidden_dim: int = 256


class FieldEncoder(nn.Module):
    def __init__(self, img_size: int, hidden: int, n_down: int, out_ch: int):
        super().__init__()
        layers: list[nn.Module] = []
        cin = 3
        s = img_size
        for i in range(n_down):
            layers += [nn.Conv2d(cin, hidden, 3, stride=2, padding=1),
                       nn.GroupNorm(8, hidden), nn.SiLU()]
            cin = hidden
            s = (s + 1) // 2
        layers += [nn.Conv2d(cin, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.SiLU(),
                   nn.Conv2d(hidden, out_ch, 3, padding=1)]
        self.net = nn.Sequential(*layers)
        self.spatial = s

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class OscField(nn.Module):

    is_syncnet = True
    has_rotors = False
    SUPPORTED_OVERRIDES = frozenset({'t_override', 'phase_override', 'return_trace'})

    GATE_OVERRIDES = frozenset(set())
    PHASE_OVERRIDES = frozenset({'freeze', 'shuffle', 'lambda0'})
    def __init__(self, cfg: OscFieldConfig, q_encoder: QuestionEncoder,
                 img_size: int, answer_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.q_encoder = q_encoder
        q_dim = q_encoder.out_dim
        d, K = cfg.osc_dim, cfg.n_groups
        self.d, self.K = d, K
        C = d * K
        self.C = C
        self.T, self.dt = cfg.T, cfg.dt

        self.encoder = FieldEncoder(img_size, cfg.hidden, cfg.n_down, cfg.field_ch)
        S = self.encoder.spatial
        self.spatial = S
        self.P = S * S
        fch = cfg.field_ch
        if cfg.q_conditioning == 'film':
            self.film_gamma = nn.Linear(q_dim, fch)
            self.film_beta = nn.Linear(q_dim, fch)
        elif cfg.q_conditioning == 'broadcast_cat':
            self.q_enc = nn.Linear(q_dim, cfg.q_emb_dim)
            fch = fch + cfg.q_emb_dim
        self.fch = fch
        self.norm = nn.GroupNorm(8, cfg.field_ch, affine=True)
        if cfg.use_pos_emb:
            self.pos_emb = nn.Parameter(0.02 * torch.randn(1, cfg.field_ch, S, S))

        # oscillator field
        self.z_head = nn.Conv2d(fch, C, 1)
        if cfg.stimulus:
            self.stim = nn.Conv2d(fch, C, 1)
        if cfg.coupling == 'conv':
            self.J = nn.Conv2d(C, C, cfg.ksize, padding=cfg.ksize // 2, bias=False)
            nn.init.normal_(self.J.weight, std=0.02)
        elif cfg.coupling != 'none':
            raise ValueError(f'unknown coupling {cfg.coupling!r}')
        # per-group antisymmetric generator, omega scales it
        raw = torch.randn(K, d, d) * 0.1
        self.omega_raw = nn.Parameter(raw) if cfg.learn_omega else None
        if not cfg.learn_omega:
            self.register_buffer('omega_fixed', raw)

        # readout
        Sl = cfg.n_slots
        self.n_slots = Sl
        self.slot_q = nn.Parameter(torch.randn(Sl, fch) / fch ** 0.5)
        self.slot_q_from_q = nn.Linear(q_dim, Sl * fch)
        self.log_beta = nn.Parameter(torch.tensor(float(np.log(cfg.beta_init))))
        if cfg.readout == 'sync':
            self.slot_q2 = nn.Parameter(torch.randn(Sl, fch) / fch ** 0.5)
            self.slot_q2_from_q = nn.Linear(q_dim, Sl * fch)
            self.log_beta2 = nn.Parameter(torch.tensor(float(np.log(cfg.beta_init))))
            self.lam = nn.Parameter(torch.tensor(float(cfg.lambda_init)))
            head_in = Sl * fch * 2 + Sl * (Sl - 1) // 2 + Sl + q_dim
        elif cfg.readout == 'content':
            head_in = Sl * fch + q_dim
        else:
            raise ValueError(f'unknown readout {cfg.readout!r}')
        self.head = nn.Sequential(nn.Linear(head_in, cfg.hidden_dim), nn.GELU(),
                                  nn.Linear(cfg.hidden_dim, answer_dim))
        iu = torch.triu_indices(Sl, Sl, offset=1)
        self.register_buffer('pair_i', iu[0]); self.register_buffer('pair_j', iu[1])

    # ------------------------------------------------------------------

    def _omega(self) -> Tensor:
        raw = self.omega_raw if self.omega_raw is not None else self.omega_fixed
        A = raw - raw.transpose(-1, -2)                             # (K, d, d)
        return A * self.cfg.omega_scale

    def _normalise(self, z: Tensor) -> Tensor:
        B, C, S, _ = z.shape
        zg = z.view(B, self.K, self.d, S, S)
        return F.normalize(zg, dim=2).view(B, C, S, S)

    def _tangent(self, z: Tensor, v: Tensor) -> Tensor:
        B, C, S, _ = z.shape
        zg = z.view(B, self.K, self.d, S, S); vg = v.view(B, self.K, self.d, S, S)
        vg = vg - (vg * zg).sum(2, keepdim=True) * zg
        return vg.reshape(B, C, S, S)

    def _rotate(self, z: Tensor) -> Tensor:
        B, C, S, _ = z.shape
        zg = z.view(B, self.K, self.d, S, S)
        out = torch.einsum('kde,bkeij->bkdij', self._omega(), zg)
        return out.reshape(B, C, S, S)

    def forward(self, images: Tensor, questions: Tensor,
                t_override: int | None = None, return_trace: bool = False,
                phase_override: str | None = None, **batch) -> dict:
        cfg = self.cfg
        q = self.q_encoder.flat(questions)
        B = images.shape[0]
        T = t_override if t_override is not None else self.T
        dev = images.device

        f = self.encoder(images)
        if cfg.q_conditioning == 'film':
            f = f * (1 + self.film_gamma(q)[..., None, None]) + self.film_beta(q)[..., None, None]
        f = self.norm(f)
        if cfg.use_pos_emb:
            f = f + self.pos_emb
        if cfg.q_conditioning == 'broadcast_cat':
            qe = self.q_enc(q)[..., None, None].expand(-1, -1, f.shape[-2], f.shape[-1])
            f = torch.cat([f, qe], 1)
        S = f.shape[-1]

        if cfg.z_init == 'feature':
            z = self._normalise(self.z_head(f))
        else:
            z = self._normalise(torch.randn(B, self.C, S, S, device=dev))
        if phase_override == 'shuffle':
            perm = torch.argsort(torch.rand(B, S * S, device=dev), dim=1)
            zf = z.flatten(2).gather(2, perm.unsqueeze(1).expand(-1, self.C, -1))
            z = zf.view(B, self.C, S, S)
        c = self.stim(f) if cfg.stimulus else None
        traces = {'field': []} if return_trace else None
        if phase_override != 'freeze':
            for _ in range(T):
                drive = torch.zeros_like(z)
                if cfg.coupling == 'conv':
                    drive = drive + self.J(z)
                if c is not None:
                    drive = drive + c
                vel = self._rotate(z) + self._tangent(z, drive)
                z = self._normalise(z + self.dt * vel)
                if return_trace:
                    traces['field'].append(z.detach())          # type: ignore

        # ---- readout ----
        X = f.flatten(2).transpose(1, 2)                            # (B, P, fch)
        Zt = z.view(B, self.K, self.d, S * S).permute(0, 3, 1, 2)   # (B, P, K, d)
        Sl = self.n_slots
        qs = F.normalize(self.slot_q.unsqueeze(0) + self.slot_q_from_q(q).view(B, Sl, -1), dim=-1)
        keys = F.normalize(X, dim=-1)
        a1 = F.softmax(self.log_beta.exp() * torch.einsum('bsc,bpc->bsp', qs, keys), dim=-1)
        f1 = torch.einsum('bsp,bpc->bsc', a1, X)
        pieces = [f1.flatten(1)]
        metrics: dict[str, float] = {}
        if cfg.readout == 'sync':
            anchor = torch.einsum('bsp,bpkd->bskd', a1, Zt)          # (B, Sl, K, d)
            anchor_n = F.normalize(anchor, dim=-1)
            align = torch.einsum('bskd,bpkd->bsp', anchor_n, Zt) / self.K   # mean over groups
            lam = self.lam * 0.0 if phase_override == 'lambda0' else self.lam
            qs2 = F.normalize(self.slot_q2.unsqueeze(0) + self.slot_q2_from_q(q).view(B, Sl, -1), dim=-1)
            l2 = lam * align + self.log_beta2.exp() * torch.einsum('bsc,bpc->bsp', qs2, keys)
            a2 = F.softmax(l2, dim=-1)
            f2 = torch.einsum('bsp,bpc->bsc', a2, X)
            coh = (a2 * align).sum(-1)                                # (B, Sl) coherence of what was read
            pair = torch.einsum('bskd,btkd->bst', anchor_n, anchor_n) / self.K
            pair = pair[:, self.pair_i, self.pair_j]                 # (B, Sl(Sl-1)/2)
            pieces += [f2.flatten(1), pair, coh]
            with torch.no_grad():
                metrics['slot_coherence'] = coh.mean().item()
                metrics['slot_pair_align'] = pair.mean().item()
                metrics['read_entropy'] = (-(a2.clamp(min=1e-12) * a2.clamp(min=1e-12).log()).sum(-1)
                                           / np.log(S * S)).mean().item()
                metrics['lambda'] = float(lam)
        pieces.append(q)
        logits = self.head(torch.cat(pieces, -1))
        with torch.no_grad():
            # global order parameter of the field (mean over groups)
            metrics['phase_R'] = Zt.mean(1).norm(dim=-1).mean().item()
        return {'logits': logits, 'traces': traces, 'metrics': metrics}
