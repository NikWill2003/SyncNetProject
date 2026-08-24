"""PhaseBind: binding by synchrony at the read, not only at the gate.

The VQA syncnet keeps a phase per *module* and uses it for one thing, the
M x M message gate. The modules' other coordination problem -- which
tokens each module binds to -- is left to content attention with shared
weights, and empirically it is not solved: without a hard partition the
modules read the same tokens (read_overlap ~ 1). This model gives the
tokens phases too and makes one oscillator system of modules and tokens:

    read        alpha_kp  propto  exp( beta cos(q_k, X_p) + lambda <z_k, z_p> )
                normalised over modules per token (tokens choose a module,
                slot-attention style) and then over tokens per module
    gate        g_ij = sigmoid( a <z_i, z_j> + b )        (modules)
    dynamics    z_k <- Pi( z_k + dt [ omega_k A z_k
                      + tangent( sum_j K_ij kappa_ij(h) z_j          module-module
                              + k_tm sum_p alpha_kp z_p ) ] )        module <- tokens it reads
                z_p <- Pi( z_p + dt [ omega_p A z_p
                      + tangent( k_tt sum_p' S_pp' z_p'              token-token (same object)
                              + k_mt sum_k w_kp z_k                  token <- modules that read it
                              + c_p ) ] )                            optional stimulus

A module pulls the tokens it reads into its phase and is pulled toward
them, so a binding, once formed, is self-reinforcing across steps; two
modules on the same object end up in phase (and communicate), modules on
different objects drift apart unless the module-module coupling kappa(h)
pulls them together for the question at hand. That is the coalition
story of Chapter 1 with the entities being tokens or objects.

Every coupling path is a config switch so the ablations remove exactly
one of them. `phase_read='none'` recovers a content-only read (the
patched syncnet with competition); `gate_mode='zero'` removes messages.

Front-ends: a grid (cnn | patchify) or the six object tokens
(object_tokens.py), in which case the tokens ARE the entities.

Runtime overrides (interventions callback):
    t_override, gate_override {open, zero, frozen, shuffle},
    phase_override {freeze, freeze_tokens, shuffle_tokens, lambda0}
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
from .encoders import build_encoder
from .object_tokens import ObjectTokenizer
from .question_encoders import QuestionEncoder
from .oscillators import (
    GateShape, HebbianCoupling, MLPCoupling, SkewGenerator, grid_neighbours,
    random_unit, sphere_step, tangent, zero_diag,
)


@dataclass
class PhaseBindConfig(ModelConfig):
    q_conditioning: str = 'film'        # film | broadcast_cat | token
    q_emb_dim: int = 32
    use_pos_emb: bool = True
    partition: str = 'none'             # none | quadrant | object
    encoder: dict[str, Any] = field(default_factory=lambda: {
        'name': 'cnn', 'ch': 128, 'hidden': 64, 'patch_size': 5})

    # modules
    n_modules: int = 6
    module_dim: int = 128
    content_dim: int = 32
    query_hidden: int = 64
    beta_init: float = 5.0
    per_module_gru: bool = True

    # oscillators
    osc_dim: int = 2
    T: int = 6
    dt: float = 0.5
    omega_scale: float = 0.5            # module natural-frequency spread
    learn_omega: bool = True
    tok_omega: str = 'feature'          # none | feature
    tok_phase_init: str = 'random'      # random | encoder
    mod_phase_init: str = 'learned'     # learned | random

    # coupling paths
    tok_coupling: str = 'local'         # local | global | none
    tok_coupling_k: float = 1.0
    mod_tok_coupling: bool = True       # module <-> tokens it reads
    mod_tok_k: float = 1.0
    mod_mod_coupling: str = 'hebbian'   # hebbian | mlp | none
    mod_mod_k: float = 1.0
    tok_stimulus: bool = False          # AKOrN stimulus c_p = W X_p on tokens

    # read
    phase_read: str = 'additive'        # additive | multiplicative | only | none
    lambda_init: float = 2.0
    learn_lambda: bool = True
    read_norm: str = 'modules'          # modules | tokens | both
    hard_assign: bool = False           # straight-through argmax over modules

    # module-module gate
    gate_mode: str = 'phase'            # phase | open | zero | attn
    gate_sharpen: bool = True
    gate_alpha_init: float = 4.0
    gate_bias_init: float = -1.0
    learn_sharpen: bool = True
    gate_zero_diag: bool = True
    k_hidden: int = 64
    msg_dim: int = 64
    msg_agg: str = 'mean'               # mean | budget | bus

    # readout
    readout_mode: str = 'sum'           # sum | concat
    hidden_dim: int = 128
    use_prior_head: bool = True
    vote_sees_q: bool = True


def _norm_entropy(p: Tensor) -> Tensor:
    n = p.shape[-1]
    ent = -(p.clamp(min=1e-12) * p.clamp(min=1e-12).log()).sum(-1)
    return ent / float(np.log(n))


class PhaseBind(nn.Module):

    is_syncnet = True
    has_rotors = False
    SUPPORTED_OVERRIDES = frozenset({'t_override', 'gate_override',
                                     'phase_override', 'return_trace'})
    GATE_OVERRIDES = frozenset({'open', 'zero', 'frozen', 'shuffle'})
    PHASE_OVERRIDES = frozenset({'freeze', 'freeze_tokens', 'shuffle_tokens', 'lambda0'})

    def __init__(self, cfg: PhaseBindConfig, q_encoder: QuestionEncoder,
                 img_size: int, answer_dim: int,
                 object_colours: list[tuple[int, int, int]] | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.q_encoder = q_encoder
        q_dim = q_encoder.out_dim
        M, dm, d = cfg.n_modules, cfg.module_dim, cfg.osc_dim
        self.M, self.dm, self.d = M, dm, d
        self.T, self.dt = cfg.T, cfg.dt
        enc = dict(cfg.encoder)
        name = enc.get('name', 'cnn')
        self.objects = name == 'objects'
        enc_ch = int(enc['ch'])
        if d < 2:
            raise ValueError('osc_dim must be >= 2')

        # ---- tokens ------------------------------------------------
        if cfg.q_conditioning == 'broadcast_cat':
            self.ch = enc_ch + cfg.q_emb_dim
            self.q_enc = nn.Linear(q_dim, cfg.q_emb_dim)
        else:
            self.ch = enc_ch
            if cfg.q_conditioning == 'film':
                self.film_gamma = nn.Linear(q_dim, enc_ch)
                self.film_beta = nn.Linear(q_dim, enc_ch)
        if self.objects:
            if object_colours is None:
                raise ValueError('objects encoder needs the task colour table')
            self.obj_tok = ObjectTokenizer(object_colours, img_size,
                                           int(enc.get('obj_size', 5)))
            self.obj_embed = nn.Linear(self.obj_tok.feat_dim, enc_ch)
            P = self.obj_tok.n_objects
            self.spatial = None
        else:
            self.encoder = build_encoder(enc, img_size)
            self.spatial = self.encoder.spatial
            P = self.spatial * self.spatial
            if cfg.use_pos_emb:
                self.pos_emb = nn.Parameter(
                    0.02 * torch.randn(1, enc_ch, self.spatial, self.spatial))
        self.P = P
        self.norm = nn.GroupNorm(8, enc_ch, affine=True)

        if cfg.partition == 'quadrant':
            if self.objects or M != 4:
                raise ValueError('quadrant partition needs a grid encoder and M=4')
            S, half = self.spatial, (self.spatial + 1) // 2
            mask = torch.zeros(M, S, S, dtype=torch.bool)
            mask[0, :half, :half] = True; mask[1, :half, half:] = True
            mask[2, half:, :half] = True; mask[3, half:, half:] = True
            self.register_buffer('partition_mask', mask.reshape(M, -1))
        elif cfg.partition == 'object':
            if not self.objects or M != P:
                raise ValueError('object partition needs the objects encoder and M=n_objects')
            self.register_buffer('partition_mask', torch.eye(M, dtype=torch.bool))
        elif cfg.partition != 'none':
            raise ValueError(f'unknown partition {cfg.partition!r}')

        # token oscillators
        if cfg.tok_phase_init == 'encoder':
            self.tok_phase_head = nn.Linear(self.ch, d)
        if cfg.tok_omega == 'feature':
            self.tok_omega_head = nn.Linear(self.ch, 1)
        if cfg.tok_stimulus:
            self.tok_stim = nn.Linear(self.ch, d)
        if cfg.tok_coupling == 'local':
            if self.objects:
                raise ValueError("tok_coupling='local' needs a grid; use 'global'")
            self.register_buffer('adj', grid_neighbours(self.spatial, eight=True))
        self.tok_sim = nn.Linear(self.ch, 32, bias=False)     # feature similarity for S_pp'
        self.log_k_tt = nn.Parameter(torch.tensor(float(np.log(cfg.tok_coupling_k))))
        self.log_k_mt = nn.Parameter(torch.tensor(float(np.log(cfg.mod_tok_k))))
        self.log_k_mm = nn.Parameter(torch.tensor(float(np.log(cfg.mod_mod_k))))

        # ---- modules -----------------------------------------------
        self.h_init = nn.Sequential(
            nn.Linear(q_dim, cfg.query_hidden), nn.GELU(),
            nn.Linear(cfg.query_hidden, M * dm))
        self.module_embed = nn.Parameter(torch.randn(M, dm) / (dm ** 0.5))
        self.attn_query = nn.Linear(dm, self.ch)
        self.log_beta = nn.Parameter(torch.tensor(float(np.log(cfg.beta_init))))
        lam = torch.tensor(float(cfg.lambda_init))
        if cfg.learn_lambda:
            self.lam = nn.Parameter(lam)
        else:
            self.register_buffer('lam', lam)
        if cfg.phase_read == 'multiplicative':
            self.read_gate = GateShape(True, 4.0, -1.0, True)
        omega0 = cfg.omega_scale * torch.linspace(-1.0, 1.0, M)
        if cfg.learn_omega:
            self.omega = nn.Parameter(omega0)
        else:
            self.register_buffer('omega', omega0)
        if cfg.mod_phase_init == 'learned':
            self.z0 = nn.Parameter(torch.randn(M, d))
        self.gen = SkewGenerator(d, learn=True)
        self.K = nn.Parameter(torch.ones(M, M))
        if cfg.mod_mod_coupling == 'hebbian':
            self.kappa = HebbianCoupling(dm, 32)
        elif cfg.mod_mod_coupling == 'mlp':
            self.kappa = MLPCoupling(dm, cfg.k_hidden)
        elif cfg.mod_mod_coupling != 'none':
            raise ValueError(f'unknown mod_mod_coupling {cfg.mod_mod_coupling!r}')

        # messages + gate
        self.msg_proj = nn.Linear(dm, cfg.msg_dim)
        if cfg.msg_agg == 'bus':
            self.bus_norm = nn.LayerNorm(cfg.msg_dim)
        self.gate_shape = GateShape(cfg.gate_sharpen, cfg.gate_alpha_init,
                                    cfg.gate_bias_init, cfg.learn_sharpen)
        if cfg.gate_mode == 'attn':
            self.gate_q = nn.Linear(dm, cfg.k_hidden)
            self.gate_k = nn.Linear(dm, cfg.k_hidden)
        elif cfg.gate_mode not in ('phase', 'open', 'zero'):
            raise ValueError(f'unknown gate_mode {cfg.gate_mode!r}')
        cell_in = self.ch + cfg.msg_dim
        if cfg.per_module_gru:
            self.cells = nn.ModuleList([nn.GRUCell(cell_in, dm) for _ in range(M)])
        else:
            self.cell = nn.GRUCell(cell_in, dm)

        # ---- readout -----------------------------------------------
        self.content_head = nn.Linear(self.ch, cfg.content_dim)
        if cfg.readout_mode == 'sum':
            vote_in = cfg.content_dim + dm + (q_dim if cfg.vote_sees_q else 0)
            self.vote_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(vote_in, cfg.hidden_dim), nn.GELU(),
                              nn.Linear(cfg.hidden_dim, answer_dim))
                for _ in range(M)])
            if cfg.use_prior_head:
                self.prior_head = nn.Sequential(
                    nn.Linear(q_dim, cfg.hidden_dim), nn.GELU(),
                    nn.Linear(cfg.hidden_dim, answer_dim))
        elif cfg.readout_mode == 'concat':
            self.head = nn.Sequential(
                nn.Linear(M * (cfg.content_dim + dm) + q_dim, cfg.hidden_dim),
                nn.GELU(), nn.Linear(cfg.hidden_dim, answer_dim))
        else:
            raise ValueError(f'unknown readout_mode {cfg.readout_mode!r}')

    # ------------------------------------------------------------------

    def _tokens(self, images: Tensor, q: Tensor) -> Tensor:
        cfg = self.cfg
        if self.objects:
            feats, found = self.obj_tok(images)
            self._last_found = found
            c = self.obj_embed(feats)                              # (B, P, ch)
            if cfg.q_conditioning == 'film':
                c = c * (1 + self.film_gamma(q)).unsqueeze(1) + self.film_beta(q).unsqueeze(1)
            c = self.norm(c.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2)
        else:
            c = self.encoder(images)                               # (B, ch, S, S)
            if cfg.q_conditioning == 'film':
                c = c * (1 + self.film_gamma(q)[..., None, None]) + self.film_beta(q)[..., None, None]
            c = self.norm(c)
            if cfg.use_pos_emb:
                c = c + self.pos_emb
            c = c.permute(0, 2, 3, 1).reshape(c.shape[0], -1, c.shape[1])
        if cfg.q_conditioning == 'broadcast_cat':
            qe = self.q_enc(q).unsqueeze(1).expand(-1, c.shape[1], -1)
            c = torch.cat([c, qe], dim=-1)
        return c

    def _token_similarity(self, X: Tensor) -> Tensor | None:
        """S_pp' in [-1, 1] (B, P, P), restricted to neighbours if local,
        row-normalised by neighbour count."""
        cfg = self.cfg
        if cfg.tok_coupling == 'none':
            return None
        k = F.normalize(self.tok_sim(X), dim=-1)
        S = torch.einsum('bpd,bqd->bpq', k, k)
        if cfg.tok_coupling == 'local':
            S = S * self.adj.unsqueeze(0)
            deg = self.adj.sum(-1).clamp(min=1.0)
            return S / deg[None, :, None]
        S = zero_diag(S)
        return S / float(max(S.shape[-1] - 1, 1))

    def _read(self, h: Tensor, X: Tensor, z_mod: Tensor, z_tok: Tensor,
              lam: Tensor, pm: Tensor | None, hard: bool):
        """-> attn (B, M, P) rows sum to 1;  assign (B, M, P) cols sum to 1."""
        cfg = self.cfg
        queries = F.normalize(self.attn_query(h), dim=-1)
        keys = F.normalize(X, dim=-1)
        content = self.log_beta.exp() * torch.einsum('bmc,bpc->bmp', queries, keys)
        phase = torch.einsum('bmd,bpd->bmp', z_mod, z_tok)
        if cfg.phase_read == 'additive':
            logits = content + lam * phase
        elif cfg.phase_read == 'only':
            logits = lam * phase
        elif cfg.phase_read == 'none':
            logits = content
        elif cfg.phase_read == 'multiplicative':
            logits = content + torch.log(self.read_gate(phase) + 1e-6) * (lam != 0).float()
        else:
            raise ValueError(f'unknown phase_read {cfg.phase_read!r}')
        if pm is not None:
            logits = logits.masked_fill(~pm.unsqueeze(0), float('-inf'))
        assign = F.softmax(logits, dim=1)                          # tokens choose modules
        if hard:
            idx = assign.argmax(1, keepdim=True)
            hard_a = torch.zeros_like(assign).scatter_(1, idx, 1.0)
            assign = hard_a + (assign - assign.detach())
        if cfg.read_norm == 'modules':
            w = assign
        elif cfg.read_norm == 'tokens':
            w = F.softmax(logits, dim=-1)
        elif cfg.read_norm == 'both':
            w = assign * F.softmax(logits, dim=-1)
        else:
            raise ValueError(f'unknown read_norm {cfg.read_norm!r}')
        if pm is not None:
            w = w.masked_fill(~pm.unsqueeze(0), 0.0)
        attn = w / (w.sum(-1, keepdim=True) + 1e-6)
        return attn, assign

    def _gate(self, h: Tensor, z_mod: Tensor) -> Tensor:
        cfg = self.cfg
        B, M, _ = h.shape
        if cfg.gate_mode == 'phase':
            g = self.gate_shape(torch.einsum('bid,bjd->bij', z_mod, z_mod))
        elif cfg.gate_mode == 'open':
            g = torch.ones(B, M, M, device=h.device)
        elif cfg.gate_mode == 'zero':
            g = torch.zeros(B, M, M, device=h.device)
        else:
            gq, gk = self.gate_q(h), self.gate_k(h)
            g = F.softmax(torch.einsum('bik,bjk->bij', gq, gk) / (gq.shape[-1] ** 0.5), -1)
        if cfg.gate_zero_diag:
            g = zero_diag(g)
        return g

    def _messages(self, g: Tensor, h: Tensor) -> Tensor:
        cfg = self.cfg
        m_all = self.msg_proj(h)
        s = torch.einsum('bij,bjk->bik', g, m_all)
        if cfg.msg_agg == 'bus':
            return self.bus_norm(s)
        if cfg.msg_agg == 'budget':
            return s / (g.sum(-1, keepdim=True) + 1.0)
        return s / (1.0 if cfg.gate_mode == 'attn' else float(self.M))

    def _update(self, a: Tensor, msg: Tensor, h: Tensor) -> Tensor:
        B, M, dm = h.shape
        inp = torch.cat([a, msg], -1)
        if self.cfg.per_module_gru:
            return torch.stack([self.cells[k](inp[:, k], h[:, k]) for k in range(M)], 1)
        return self.cell(inp.reshape(B * M, -1), h.reshape(B * M, dm)).reshape(B, M, dm)

    # ------------------------------------------------------------------

    def forward(self, images: Tensor, questions: Tensor,
                t_override: int | None = None, return_trace: bool = False,
                gate_override: str | None = None,
                phase_override: str | None = None, **batch) -> dict:
        cfg = self.cfg
        q = self.q_encoder.flat(questions)
        B, M, dm, d = images.shape[0], self.M, self.dm, self.d
        T = t_override if t_override is not None else self.T
        dev = images.device

        X = self._tokens(images, q)                                # (B, P, ch)
        P = X.shape[1]
        pm = getattr(self, 'partition_mask', None)
        S = self._token_similarity(X)
        b_content = self.content_head(X)

        # module state
        h = self.h_init(q).reshape(B, M, dm) + self.module_embed.unsqueeze(0)
        if cfg.mod_phase_init == 'learned':
            z_mod = F.normalize(self.z0, dim=-1).unsqueeze(0).expand(B, M, d)
        else:
            z_mod = random_unit(B, M, d, device=dev)
        # token state
        if cfg.tok_phase_init == 'encoder':
            z_tok = F.normalize(self.tok_phase_head(X), dim=-1)
        else:
            z_tok = random_unit(B, P, d, device=dev)
        if phase_override == 'shuffle_tokens':
            perm = torch.argsort(torch.rand(B, P, device=dev), dim=1)
            z_tok = z_tok.gather(1, perm.unsqueeze(-1).expand_as(z_tok))
        omega_tok = (torch.tanh(self.tok_omega_head(X)).squeeze(-1) * cfg.omega_scale
                     if cfg.tok_omega == 'feature' else torch.zeros(B, P, device=dev))
        stim = tangent(z_tok, self.tok_stim(X)) if cfg.tok_stimulus else None
        lam = self.lam * 0.0 if phase_override == 'lambda0' else self.lam
        k_tt, k_mt, k_mm = self.log_k_tt.exp(), self.log_k_mt.exp(), self.log_k_mm.exp()
        evolve_mod = phase_override not in ('freeze',) and gate_override != 'frozen'
        evolve_tok = phase_override not in ('freeze', 'freeze_tokens')

        attn = torch.full((B, M, P), 1.0 / P, device=dev)
        assign = torch.full((B, M, P), 1.0 / M, device=dev)
        g = self._gate(h, z_mod)
        g_held = g if gate_override == 'frozen' else None
        g_hist, traces = [], ({'phase': [], 'tok_phase': [], 'gates': [], 'attn': [], 'h': []}
                              if return_trace else None)
        for _ in range(T):
            attn, assign = self._read(h, X, z_mod, z_tok, lam, pm, cfg.hard_assign)
            a = torch.einsum('bmp,bpc->bmc', attn, X)
            g = g_held if g_held is not None else self._gate(h, z_mod)
            if gate_override == 'open':
                g = torch.ones(B, M, M, device=dev)
            elif gate_override == 'zero':
                g = torch.zeros(B, M, M, device=dev)
            elif gate_override == 'shuffle':
                g = g[torch.randperm(B, device=dev)]
            g_hist.append(g)
            msg = self._messages(g, h)
            h = self._update(a, msg, h)

            # ---- one phase step for the whole system ----
            if evolve_mod:
                vel = self.gen(z_mod, self.omega.unsqueeze(0).expand(B, M))
                pull = torch.zeros_like(z_mod)
                if cfg.mod_mod_coupling != 'none':
                    w = self.K.unsqueeze(0) * self.kappa(h) * k_mm
                    pull = pull + torch.einsum('bij,bjd->bid', w, z_mod)
                if cfg.mod_tok_coupling:
                    pull = pull + k_mt * torch.einsum('bmp,bpd->bmd', attn, z_tok)
                z_mod_new = sphere_step(z_mod, vel + tangent(z_mod, pull), self.dt)
            else:
                z_mod_new = z_mod
            if evolve_tok:
                vel = self.gen(z_tok, omega_tok)
                pull = torch.zeros_like(z_tok)
                if S is not None:
                    pull = pull + k_tt * torch.einsum('bpq,bqd->bpd', S, z_tok)
                if cfg.mod_tok_coupling:
                    pull = pull + k_mt * torch.einsum('bmp,bmd->bpd', assign, z_mod)
                vel = vel + tangent(z_tok, pull)
                if stim is not None:
                    vel = vel + stim
                z_tok = sphere_step(z_tok, vel, self.dt)
            z_mod = z_mod_new

            if return_trace:
                traces['phase'].append(z_mod.detach()); traces['tok_phase'].append(z_tok.detach())  # type: ignore
                traces['gates'].append(g.detach()); traces['attn'].append(attn.detach()); traces['h'].append(h.detach())  # type: ignore

        m_content = torch.einsum('bmp,bpd->bmd', attn, b_content)
        if cfg.readout_mode == 'sum':
            parts = ([m_content[:, k], h[:, k]] + ([q] if cfg.vote_sees_q else []) for k in range(M))
            module_logits = torch.stack(
                [self.vote_heads[k](torch.cat(p, -1)) for k, p in enumerate(parts)], 1)
            prior = self.prior_head(q) if cfg.use_prior_head else torch.zeros_like(module_logits[:, 0])
            logits = prior + module_logits.sum(1)
        else:
            logits = self.head(torch.cat([m_content.flatten(1), h.flatten(1), q], -1))

        with torch.no_grad():
            off = ~torch.eye(M, dtype=torch.bool, device=dev)
            metrics = {
                'phase_R': z_mod.mean(1).norm(dim=-1).mean().item(),
                'tok_R': z_tok.mean(1).norm(dim=-1).mean().item(),
                'gate_offdiag': g[:, off].mean().item(),
                'gate_entropy': _norm_entropy(g / (g.sum(-1, keepdim=True) + 1e-6)).mean().item(),
                'read_entropy': _norm_entropy(attn).mean().item(),
                'assign_purity': assign.max(1).values.mean().item(),
                'module_use': _norm_entropy(assign.sum(-1) / P).mean().item(),
                # how in-phase are the tokens a module reads? (binding by phase)
                'bind_R': torch.einsum('bmp,bpd->bmd', attn, z_tok).norm(dim=-1).mean().item(),
                'lambda': float(lam),
            }
            an = F.normalize(attn, dim=-1)
            metrics['read_overlap'] = torch.einsum('bmp,bnp->bmn', an, an)[:, off].mean().item()
            if len(g_hist) > 1:
                metrics['gate_tvar'] = torch.stack(g_hist).var(0, unbiased=False)[:, off].mean().item()
            if self.objects:
                metrics['obj_found'] = self._last_found.float().sum(-1).mean().item()
        return {'logits': logits, 'traces': traces, 'metrics': metrics}
