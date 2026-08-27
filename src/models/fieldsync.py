"""FieldSync: modules over an oscillator field.

The screen showed two things: a phase variable used as a module-module
gate collapses to global synchrony under every regime, and a phase field
over the feature map (OscField) carries relational computation causally
(+.15 ternary over a content readout; freezing the phases costs .56). This
model keeps the field and puts the modules back on top of it.

    field     z_p  <- OscillatorField(X)                     T_field steps
    anchor    alpha1_kp ~ exp(beta cos(q1(h_k), X_p)),  phi_k = norm(sum_p alpha1_kp z_p)
    read      alpha2_kp ~ exp(lam <phi_k, z_p> + beta' cos(q2(h_k), X_p))
              a_k = sum_p alpha2_kp X_p,   c_k = sum_p alpha2_kp <phi_k, z_p>
    gate      g_ij = sigmoid(a <phi_i, phi_j> + b)           read off the field
    update    h_k <- GRU_k([a_k; c_k; msg_k], h_k)

Two design decisions follow from the screen. lam is FIXED (every learnable
phase weight went to zero), so the read-by-phase path cannot be switched
off by the optimiser; lam=0 is the content-only ablation. And the gate is
not an evolved variable but a readout of the field: two modules are
coupled when their anchors' phases have propagated into each other, i.e.
when the objects they hold are geometrically related. There is nothing
for the optimiser to collapse, and the gate carries information that the
attention gate would have to compute from positions.

gate_mode: field | attn | open | zero.
Overrides: t_override (module steps), gate_override {open, zero, frozen,
shuffle}, phase_override {freeze (no field dynamics), shuffle (permute
field positions), lambda0}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.config import ModelConfig
from .question_encoders import QuestionEncoder
from .osc_core import FieldEncoder, OscillatorField
from .oscillators import GateShape, zero_diag


@dataclass
class FieldSyncConfig(ModelConfig):
    q_conditioning: str = 'film'        # film | broadcast_cat | token
    q_emb_dim: int = 32
    # field (OscField defaults)
    hidden: int = 48
    n_down: int = 2
    field_ch: int = 64
    use_pos_emb: bool = True
    osc_dim: int = 4
    n_groups: int = 16
    T_field: int = 8
    dt: float = 1.0
    ksize: int = 5
    coupling: str = 'conv'              # conv | none
    stimulus: bool = True
    learn_omega: bool = True
    omega_scale: float = 0.1
    z_init: str = 'feature'             # feature | random
    # modules
    n_modules: int = 4
    module_dim: int = 128
    T: int = 4
    per_module_gru: bool = True
    beta_init: float = 5.0
    lam: float = 4.0                    # FIXED phase weight in the read; 0 = content-only
    gate_mode: str = 'field'            # field | attn | open | zero
    gate_alpha_init: float = 4.0
    gate_bias_init: float = -1.0
    gate_zero_diag: bool = True
    msg_dim: int = 64
    k_hidden: int = 64
    # readout
    content_dim: int = 32
    readout_mode: str = 'sum'           # sum | concat
    hidden_dim: int = 128
    use_prior_head: bool = True
    vote_sees_q: bool = True


def _norm_entropy(p: Tensor) -> Tensor:
    n = p.shape[-1]
    return -(p.clamp(min=1e-12) * p.clamp(min=1e-12).log()).sum(-1) / float(np.log(n))


class FieldSync(nn.Module):

    is_syncnet = True
    has_rotors = False
    objects = False
    SUPPORTED_OVERRIDES = frozenset({'t_override', 'gate_override', 'phase_override', 'return_trace'})
    GATE_OVERRIDES = frozenset({'open', 'zero', 'frozen', 'shuffle'})
    PHASE_OVERRIDES = frozenset({'freeze', 'shuffle', 'lambda0'})

    def __init__(self, cfg: FieldSyncConfig, q_encoder: QuestionEncoder,
                 img_size: int, answer_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.q_encoder = q_encoder
        q_dim = q_encoder.out_dim
        M, dm = cfg.n_modules, cfg.module_dim
        self.M, self.dm, self.T = M, dm, cfg.T

        # ---- features + field ----
        self.encoder = FieldEncoder(img_size, cfg.hidden, cfg.n_down, cfg.field_ch)
        S = self.encoder.spatial
        self.spatial, self.P = S, S * S
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
        self.field = OscillatorField(fch, cfg.osc_dim, cfg.n_groups, cfg.T_field, cfg.dt,
                                     cfg.ksize, cfg.coupling, cfg.stimulus, cfg.learn_omega,
                                     cfg.omega_scale, cfg.z_init)
        self.K, self.d = self.field.K, self.field.d

        # ---- modules ----
        self.h_init = nn.Sequential(nn.Linear(q_dim, 64), nn.GELU(), nn.Linear(64, M * dm))
        self.module_embed = nn.Parameter(torch.randn(M, dm) / dm ** 0.5)
        self.q1 = nn.Linear(dm, fch)              # anchor query
        self.q2 = nn.Linear(dm, fch)              # content term of the phase read
        self.log_beta = nn.Parameter(torch.tensor(float(np.log(cfg.beta_init))))
        self.log_beta2 = nn.Parameter(torch.tensor(float(np.log(cfg.beta_init))))
        self.register_buffer('lam', torch.tensor(float(cfg.lam)))
        self.gate_shape = GateShape(True, cfg.gate_alpha_init, cfg.gate_bias_init, True)
        if cfg.gate_mode == 'attn':
            self.gate_q = nn.Linear(dm, cfg.k_hidden)
            self.gate_k = nn.Linear(dm, cfg.k_hidden)
        elif cfg.gate_mode not in ('field', 'open', 'zero'):
            raise ValueError(f'unknown gate_mode {cfg.gate_mode!r}')
        self.msg_proj = nn.Linear(dm, cfg.msg_dim)
        cell_in = fch + 1 + cfg.msg_dim
        if cfg.per_module_gru:
            self.cells = nn.ModuleList([nn.GRUCell(cell_in, dm) for _ in range(M)])
        else:
            self.cell = nn.GRUCell(cell_in, dm)

        # ---- readout ----
        self.content_head = nn.Linear(fch, cfg.content_dim)
        if cfg.readout_mode == 'sum':
            vote_in = cfg.content_dim + 1 + dm + (q_dim if cfg.vote_sees_q else 0)
            self.vote_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(vote_in, cfg.hidden_dim), nn.GELU(),
                              nn.Linear(cfg.hidden_dim, answer_dim)) for _ in range(M)])
            if cfg.use_prior_head:
                self.prior_head = nn.Sequential(nn.Linear(q_dim, cfg.hidden_dim), nn.GELU(),
                                                nn.Linear(cfg.hidden_dim, answer_dim))
        elif cfg.readout_mode == 'concat':
            self.head = nn.Sequential(
                nn.Linear(M * (cfg.content_dim + 1 + dm) + q_dim, cfg.hidden_dim), nn.GELU(),
                nn.Linear(cfg.hidden_dim, answer_dim))
        else:
            raise ValueError(f'unknown readout_mode {cfg.readout_mode!r}')

    # ------------------------------------------------------------------

    def _features(self, images: Tensor, q: Tensor) -> Tensor:
        cfg = self.cfg
        f = self.encoder(images)
        if cfg.q_conditioning == 'film':
            f = f * (1 + self.film_gamma(q)[..., None, None]) + self.film_beta(q)[..., None, None]
        f = self.norm(f)
        if cfg.use_pos_emb:
            f = f + self.pos_emb
        if cfg.q_conditioning == 'broadcast_cat':
            qe = self.q_enc(q)[..., None, None].expand(-1, -1, f.shape[-2], f.shape[-1])
            f = torch.cat([f, qe], 1)
        return f

    def _gate(self, h: Tensor, phi: Tensor) -> Tensor:
        cfg = self.cfg
        B, M, _ = h.shape
        if cfg.gate_mode == 'field':
            dots = torch.einsum('bikd,bjkd->bij', phi, phi) / self.K
            g = self.gate_shape(dots)
        elif cfg.gate_mode == 'attn':
            gq, gk = self.gate_q(h), self.gate_k(h)
            g = F.softmax(torch.einsum('bik,bjk->bij', gq, gk) / gq.shape[-1] ** 0.5, -1)
        elif cfg.gate_mode == 'open':
            g = torch.ones(B, M, M, device=h.device, dtype=h.dtype)
        else:
            g = torch.zeros(B, M, M, device=h.device, dtype=h.dtype)
        return zero_diag(g) if cfg.gate_zero_diag else g

    def _update(self, inp: Tensor, h: Tensor) -> Tensor:
        B, M, dm = h.shape
        if self.cfg.per_module_gru:
            return torch.stack([self.cells[k](inp[:, k], h[:, k]) for k in range(M)], 1)
        return self.cell(inp.reshape(B * M, -1), h.reshape(B * M, dm)).reshape(B, M, dm)

    # ------------------------------------------------------------------

    def forward(self, images: Tensor, questions: Tensor,
                t_override: int | None = None, return_trace: bool = False,
                gate_override: str | None = None, phase_override: str | None = None,
                **batch) -> dict:
        cfg = self.cfg
        q = self.q_encoder.flat(questions)
        B, M = images.shape[0], self.M
        T = t_override if t_override is not None else self.T
        dev = images.device

        f = self._features(images, q)                                  # (B, fch, S, S)
        z = self.field(f, freeze=phase_override == 'freeze', shuffle=phase_override == 'shuffle')
        Zt = self.field.to_tokens(z)                                    # (B, P, K, d)
        X = f.flatten(2).transpose(1, 2)                                # (B, P, fch)
        keys = F.normalize(X, dim=-1)
        b_content = self.content_head(X)
        lam = self.lam * 0.0 if phase_override == 'lambda0' else self.lam

        h = self.h_init(q).reshape(B, M, self.dm) + self.module_embed.unsqueeze(0)
        P = X.shape[1]
        attn = torch.full((B, M, P), 1.0 / P, device=dev, dtype=X.dtype)
        coh = torch.zeros(B, M, device=dev, dtype=X.dtype)
        phi = F.normalize(torch.einsum('bmp,bpkd->bmkd', attn, Zt), dim=-1)
        g = self._gate(h, phi)
        g_held = g if gate_override == 'frozen' else None
        g_hist: list[Tensor] = []
        traces = ({'phase': [], 'gates': [], 'attn': [], 'h': [], 'field': [z.detach()]}
                  if return_trace else None)
        for _ in range(T):
            # anchor by content
            q1 = F.normalize(self.q1(h), dim=-1)
            a1 = F.softmax(self.log_beta.exp() * torch.einsum('bmc,bpc->bmp', q1, keys), -1)
            phi = F.normalize(torch.einsum('bmp,bpkd->bmkd', a1, Zt), dim=-1)     # (B, M, K, d)
            # read by phase (+ content)
            align = torch.einsum('bmkd,bpkd->bmp', phi, Zt) / self.K              # (B, M, P)
            q2 = F.normalize(self.q2(h), dim=-1)
            attn = F.softmax(lam * align + self.log_beta2.exp() * torch.einsum('bmc,bpc->bmp', q2, keys), -1)
            a = torch.einsum('bmp,bpc->bmc', attn, X)
            coh = (attn * align).sum(-1)                                            # (B, M)
            # gate from the field, messages, update
            g = g_held if g_held is not None else self._gate(h, phi)
            if gate_override == 'open':
                g = zero_diag(torch.ones(B, M, M, device=dev, dtype=h.dtype))
            elif gate_override == 'zero':
                g = torch.zeros(B, M, M, device=dev, dtype=h.dtype)
            elif gate_override == 'shuffle':
                g = g[torch.randperm(B, device=dev)]
            g_hist.append(g)
            msg = torch.einsum('bij,bjk->bik', g, self.msg_proj(h)) / float(M)
            h = self._update(torch.cat([a, coh.unsqueeze(-1), msg], -1), h)
            if return_trace:
                traces['phase'].append(F.normalize(phi.detach().flatten(2), dim=-1)); traces['gates'].append(g.detach())  # type: ignore
                traces['attn'].append(attn.detach()); traces['h'].append(h.detach())                # type: ignore

        m_content = torch.einsum('bmp,bpd->bmd', attn, b_content)
        if cfg.readout_mode == 'sum':
            parts = ([m_content[:, k], coh[:, k:k + 1], h[:, k]] + ([q] if cfg.vote_sees_q else [])
                     for k in range(M))
            module_logits = torch.stack([self.vote_heads[k](torch.cat(p, -1)) for k, p in enumerate(parts)], 1)
            prior = self.prior_head(q) if cfg.use_prior_head else torch.zeros_like(module_logits[:, 0])
            logits = prior + module_logits.sum(1)
        else:
            logits = self.head(torch.cat([m_content.flatten(1), coh, h.flatten(1), q], -1))

        with torch.no_grad():
            off = ~torch.eye(M, dtype=torch.bool, device=dev)
            phin = phi.flatten(2)
            pair = torch.einsum('bikd,bjkd->bij', phi, phi) / self.K
            an = F.normalize(attn, dim=-1)
            metrics = {
                'phase_R': Zt.flatten(2).mean(1).norm(dim=-1).mean().item() / (self.K ** 0.5),
                'slot_coherence': coh.mean().item(),
                'anchor_pair_align': pair[:, off].mean().item(),
                'gate_offdiag': g[:, off].mean().item(),
                'gate_entropy': _norm_entropy(g / (g.sum(-1, keepdim=True) + 1e-6)).mean().item(),
                'read_entropy': _norm_entropy(attn).mean().item(),
                'read_overlap': torch.einsum('bmp,bnp->bmn', an, an)[:, off].mean().item(),
                'lambda': float(lam),
            }
            if len(g_hist) > 1:
                metrics['gate_tvar'] = torch.stack(g_hist).var(0, unbiased=False)[:, off].mean().item()
        return {'logits': logits, 'traces': traces, 'metrics': metrics}
