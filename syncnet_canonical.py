"""The canonical SyncNet in the minimal file's form: same class names, same config
fields, same `from_config(cfg, dataset, answer_dim)` and `forward(batch) -> {'logits'}`.

What differs from syncnet_min.py is exactly what the thesis model has and the
minimal one drops: an oscillator field on the stem (binding by phase), per-module
anchor priors, three anchor-refining competition rounds, a question that also
FiLMs the stem and the slot contents, natural-frequency rotation in the phase
step, and a learned signed pairwise coupling tanh(MLP[h_k; h_j]).

Parameter names match src/models/sync/identity_busnet.py, so a checkpoint from
the main codebase loads with strict=True (question_encoder 'identity' on
Sort-of-CLEVR, 'onehot' on SQOOP). An 'lstm' question encoder builds a variant.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.common.img_enc import build_image_encoder
from src.models.common.qst_enc import build_question_encoder

FIELD_CH, FIELD_HIDDEN, FIELD_GROUPS, FIELD_OSC_D, T_FIELD, DT_FIELD, TOK_DIM, BETA = 64, 48, 16, 4, 8, 1.0, 64, 8.0
IMG = {'sort_of_clevr': (75, 18, 10), 'sqoop': (64, 3, 2)}          # image size, raw question size, answers


@dataclass
class SyncNetConfig:
    image_encoder: dict = field(default_factory=lambda: {'name': 'field'})
    question_encoder: dict = field(default_factory=lambda: {'name': 'identity'})   # identity | onehot | lstm

    n_modules: int = 6
    phase_dim: int = 6
    n_steps: int = 8
    phase_step_size: float = 0.5

    slot_dim: int = TOK_DIM            # fixed by the field interface; kept for parity
    slot_key_dim: int = FIELD_GROUPS * FIELD_OSC_D
    slot_bias_init: str = 'zero'       # zero | partition | random
    slot_bias_scale: float = 4.0
    competition_rounds: int = 3

    module_dim: int = 96
    message_dim: int = 4
    kappa_dim: int = 64
    readout_hidden_dim: int = 128

    medium: str = 'bus'                # bus | silent | attention
    addresses: str = 'computed'        # computed | static


# ---------------------------------------------------------------- perception
class FieldEncoder(nn.Module):
    def __init__(self, img_size: int) -> None:
        super().__init__()
        layers, cin, s = [], 3, img_size
        for _ in range(2):
            layers += [nn.Conv2d(cin, FIELD_HIDDEN, 3, stride=2, padding=1), nn.GroupNorm(8, FIELD_HIDDEN), nn.SiLU()]
            cin, s = FIELD_HIDDEN, (s + 1) // 2
        layers += [nn.Conv2d(cin, FIELD_HIDDEN, 3, padding=1), nn.GroupNorm(8, FIELD_HIDDEN), nn.SiLU(), nn.Conv2d(FIELD_HIDDEN, FIELD_CH, 3, padding=1)]
        self.net, self.spatial = nn.Sequential(*layers), s

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AdaptedTrunk(nn.Module):
    """Any codebase stem adapted to the field interface by a 1x1 conv to FIELD_CH."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base, self.adapt, self.spatial = base, nn.Conv2d(base.ch, FIELD_CH, 1), base.spatial

    def forward(self, x: Tensor) -> Tensor:
        return self.adapt(self.base(x))


class OscillatorField(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.d, self.K, self.C, self.T, self.dt = FIELD_OSC_D, FIELD_GROUPS, FIELD_OSC_D * FIELD_GROUPS, T_FIELD, DT_FIELD
        self.z_head, self.stim = nn.Conv2d(FIELD_CH, self.C, 1), nn.Conv2d(FIELD_CH, self.C, 1)
        self.J = nn.Conv2d(self.C, self.C, 5, padding=2, bias=False)
        nn.init.normal_(self.J.weight, std=0.02)
        self.omega_raw = nn.Parameter(torch.randn(FIELD_GROUPS, FIELD_OSC_D, FIELD_OSC_D) * 0.1)
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
        return z.view(B, self.K, self.d, S * S).permute(0, 3, 1, 2)

    def forward(self, f: Tensor) -> Tensor:
        z, c = self.normalise(self.z_head(f)), self.stim(f)
        for _ in range(self.T):
            z = self.normalise(z + self.dt * (self.rotate(z) + self.tangent(z, self.J(z) + c)))
        return z


class SeededCompetition(nn.Module):
    """Cells choose one slot each by PHASE alignment with a sampled per-slot anchor,
    seeded by the spatial bias; anchors refine over `rounds`."""

    def __init__(self, n_slots: int, n_cells: int, bias_init: str, bias_scale: float, rounds: int = 3) -> None:
        super().__init__()
        self.n_slots, self.iters = n_slots, rounds
        bias = torch.zeros(n_slots, n_cells)
        S = int(round(n_cells ** 0.5))
        if bias_init == 'partition':
            rows = max(1, int(n_slots ** 0.5)); cols = -(-n_slots // rows)
            ys = torch.arange(S).repeat_interleave(S); xs = torch.arange(S).repeat(S)
            region = (ys * rows // S) * cols + (xs * cols // S)
            for k in range(min(n_slots, rows * cols)):
                bias[k, region == k] = bias_scale
        elif bias_init == 'random':
            region = torch.randint(0, n_slots, (n_cells,), generator=torch.Generator().manual_seed(0))
            for k in range(n_slots):
                bias[k, region == k] = bias_scale
        elif bias_init != 'zero':
            raise ValueError(f'unknown slot_bias_init {bias_init!r}')
        self.cell_bias = nn.Parameter(bias)
        self.register_buffer('prior_scale', torch.ones(()))
        self.anchor_mu = nn.Parameter(torch.randn(1, n_slots, FIELD_GROUPS, FIELD_OSC_D))
        self.anchor_log_sigma = nn.Parameter(torch.zeros(1, n_slots, FIELD_GROUPS, FIELD_OSC_D))
        self.log_beta = nn.Parameter(torch.log(torch.tensor(BETA)))
        self.to_slot = nn.Sequential(nn.LayerNorm(FIELD_CH), nn.Linear(FIELD_CH, TOK_DIM))

    def forward(self, feats: Tensor, Zt: Tensor) -> tuple[Tensor, Tensor]:
        B, K_f = feats.shape[0], Zt.shape[2]
        eps = torch.randn(B, self.n_slots, FIELD_GROUPS, FIELD_OSC_D, device=feats.device, dtype=feats.dtype)
        phi = F.normalize(self.anchor_mu + self.anchor_log_sigma.exp() * eps, dim=-1)
        for _ in range(self.iters):
            logits = self.log_beta.exp() * torch.einsum('bkgd,bngd->bkn', phi, Zt) / K_f + self.prior_scale * self.cell_bias
            reads = F.softmax(logits, dim=1)
            reads = reads / (reads.sum(-1, keepdim=True) + 1e-8)
            phi = F.normalize(torch.einsum('bkn,bngd->bkgd', reads, Zt), dim=-1)
            slots = self.to_slot(torch.einsum('bkn,bnf->bkf', reads, feats))
        return slots, phi


class QuestionPathways(nn.Module):
    def __init__(self, q_size: int, n_modules: int, module_dim: int) -> None:
        super().__init__()
        self.grid_film_gamma, self.grid_film_beta = nn.Linear(q_size, FIELD_CH), nn.Linear(q_size, FIELD_CH)
        self.grid_norm = nn.GroupNorm(8, FIELD_CH, affine=True)
        self.film_gamma, self.film_beta, self.norm = nn.Linear(q_size, TOK_DIM), nn.Linear(q_size, TOK_DIM), nn.LayerNorm(TOK_DIM)
        self.h_init = nn.Sequential(nn.Linear(q_size, 64), nn.GELU(), nn.Linear(64, n_modules * module_dim))
        self.head_init = nn.Sequential(nn.Linear(q_size, 64), nn.GELU(), nn.Linear(64, module_dim))
        self.head_embed = nn.Parameter(torch.randn(1, module_dim) / module_dim ** 0.5)
        self.n_modules, self.module_dim = n_modules, module_dim

    def encoder_film(self, f: Tensor, q: Tensor, pos_emb: Tensor) -> Tensor:
        return self.grid_norm(f * (1 + self.grid_film_gamma(q))[..., None, None] + self.grid_film_beta(q)[..., None, None]) + pos_emb

    def content_film(self, X: Tensor, q: Tensor) -> Tensor:
        return self.norm(X * (1 + self.film_gamma(q)).unsqueeze(1) + self.film_beta(q).unsqueeze(1))

    def init_states(self, q: Tensor) -> tuple[Tensor, Tensor]:
        return self.h_init(q).reshape(q.shape[0], self.n_modules, self.module_dim), self.head_init(q).unsqueeze(1) + self.head_embed.unsqueeze(0)


# ---------------------------------------------------------------- communication
class Frame:
    def __init__(self, phase_dim: int, seed: int = 1234) -> None:
        g = torch.Generator().manual_seed(seed)
        self.ref = F.normalize(torch.randn(phase_dim - 1, phase_dim, generator=g), dim=-1)

    def __call__(self, z: Tensor) -> Tensor:
        vecs = [z]
        for k in range(z.shape[-1] - 1):
            v = self.ref[k].to(z).expand_as(z)
            for u in vecs:
                v = v - (v * u).sum(-1, keepdim=True) * u
            vecs.append(F.normalize(v, dim=-1))
        return torch.stack(vecs, 2)


class Bus(nn.Module):
    def __init__(self, module_dim: int, message_dim: int, phase_dim: int) -> None:
        super().__init__()
        self.msg_proj = nn.Linear(module_dim, message_dim)
        self.frame = Frame(phase_dim)
        self.register_buffer('frame_ref', self.frame.ref.clone())
        self.out_dim = message_dim * phase_dim

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        m = self.msg_proj(h)
        bus = torch.einsum('bnD,bnd->bDd', m, z)
        Q = self.frame(z)
        r = torch.einsum('bDd,bnad->bnDa', bus, Q) - torch.einsum('bnD,bnd,bnad->bnDa', m, z, Q)
        return r.flatten(2) / h.shape[1]


class Silent(nn.Module):
    def __init__(self, module_dim: int, message_dim: int, phase_dim: int) -> None:
        super().__init__()
        self.out_dim = message_dim * phase_dim

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        return h.new_zeros(h.shape[0], h.shape[1], self.out_dim)


class Attention(nn.Module):
    def __init__(self, module_dim: int, message_dim: int, phase_dim: int) -> None:
        super().__init__()
        self.msg_proj = nn.Linear(module_dim, message_dim)
        self.q, self.k = nn.Linear(module_dim, module_dim), nn.Linear(module_dim, module_dim)
        self.out_dim = message_dim

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        N = h.shape[1]
        att = torch.einsum('bnd,bmd->bnm', self.q(h), self.k(h)) / h.shape[-1] ** 0.5
        att = att.masked_fill(torch.eye(N, dtype=torch.bool, device=h.device), float('-inf')).softmax(-1)
        return torch.einsum('bnm,bmD->bnD', att, self.msg_proj(h))


class PhaseStep(nn.Module):
    """z_k <- Pi( z_k + gamma( omega_k A z_k + Proj_z( sum_j J_kj tanh(MLP[h_k;h_j]) z_j + W_c h_k ) ) )."""

    def __init__(self, n_rows: int, module_dim: int, phase_dim: int, kappa_dim: int, dt: float) -> None:
        super().__init__()
        self.dt = dt
        self.omega = nn.Parameter(torch.zeros(n_rows))
        self.K = nn.Parameter(torch.ones(n_rows, n_rows))
        self.k_mlp = nn.Sequential(nn.Linear(2 * module_dim, kappa_dim), nn.GELU(), nn.Linear(kappa_dim, 1), nn.Tanh())
        self.stim = nn.Linear(module_dim, phase_dim)
        self.gen = nn.Module()
        self.gen.raw = nn.Parameter(0.1 * torch.randn(phase_dim, phase_dim))

    def forward(self, z: Tensor, h: Tensor) -> Tensor:
        B, N, dm = h.shape
        A = self.gen.raw - self.gen.raw.t()
        A = A / (A.norm() / math.sqrt(2) + 1e-6)
        vel = self.omega.to(z.dtype)[None, :, None] * torch.einsum('de,bne->bnd', A, z)
        kap = self.k_mlp(torch.cat([h.unsqueeze(2).expand(B, N, N, dm), h.unsqueeze(1).expand(B, N, N, dm)], -1)).squeeze(-1)
        pull = torch.einsum('bij,bjd->bid', self.K.to(z.dtype).unsqueeze(0) * kap, z)
        tangent = lambda v: v - (v * z).sum(-1, keepdim=True) * z
        return F.normalize(z + self.dt * (vel + tangent(pull) + tangent(self.stim(h))), dim=-1)


class PrivateCells(nn.Module):
    def __init__(self, n_rows: int, in_dim: int, module_dim: int, n_modules: int) -> None:
        super().__init__()
        self.cells = nn.ModuleList(nn.GRUCell(in_dim, module_dim) for _ in range(n_rows))
        self.embeds = nn.Parameter(torch.randn(n_modules, module_dim) / module_dim ** 0.5)

    def step(self, inp: Tensor, h: Tensor) -> Tensor:
        return torch.stack([cell(inp[:, k], h[:, k]) for k, cell in enumerate(self.cells)], 1)


class HeadReadout(nn.Module):
    def __init__(self, module_dim: int, q_size: int, answer_dim: int, hidden: int) -> None:
        super().__init__()
        self.head_out = nn.Sequential(nn.Linear(module_dim + q_size, hidden), nn.GELU(), nn.Linear(hidden, answer_dim))

    def forward(self, h_head: Tensor, q: Tensor) -> Tensor:
        return self.head_out(torch.cat([h_head, q], -1))


# ---------------------------------------------------------------- the model
class SyncNet(nn.Module):
    MEDIA = {'bus': Bus, 'silent': Silent, 'attention': Attention}

    def __init__(self, cfg: SyncNetConfig, dataset: str, answer_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        img_size, q_raw, _ = IMG[dataset]
        enc = cfg.image_encoder.get('name', 'field')
        self.field_enc = FieldEncoder(img_size) if enc == 'field' else AdaptedTrunk(build_image_encoder(dict(cfg.image_encoder), dataset))
        qname = cfg.question_encoder.get('name', 'identity')
        if qname in ('identity', 'onehot'):
            self.question_encoder = None
            self.q_vocab = cfg.question_encoder.get('vocab', 40) if qname == 'onehot' else None
            self.q_size = q_raw * (self.q_vocab or 1)
        else:
            self.question_encoder = build_question_encoder(dict(cfg.question_encoder), dataset, allowed={'identity', 'mlp', 'lstm'})
            self.q_vocab, self.q_size = None, self.question_encoder.output_shape[0]
        self.M, self.N, self.d, self.T = cfg.n_modules, cfg.n_modules + 1, cfg.phase_dim, cfg.n_steps

        S = self.field_enc.spatial
        self.pos_emb = nn.Parameter(0.02 * torch.randn(1, FIELD_CH, S, S))
        self.field = OscillatorField()
        self.binder = SeededCompetition(cfg.n_modules, S * S, cfg.slot_bias_init, cfg.slot_bias_scale, cfg.competition_rounds)
        self.anchor_to_phase = nn.Linear(FIELD_GROUPS * FIELD_OSC_D, cfg.phase_dim)
        self.pathways = QuestionPathways(self.q_size, cfg.n_modules, cfg.module_dim)
        self.medium = self.MEDIA[cfg.medium](cfg.module_dim, cfg.message_dim, cfg.phase_dim)
        self.identity = PrivateCells(self.N, TOK_DIM + self.medium.out_dim, cfg.module_dim, cfg.n_modules)
        self.dynamics = PhaseStep(self.N, cfg.module_dim, cfg.phase_dim, cfg.kappa_dim, cfg.phase_step_size)
        self.readout = HeadReadout(cfg.module_dim, self.q_size, answer_dim, cfg.readout_hidden_dim)
        if cfg.addresses == 'static':
            self.z_static = nn.Parameter(F.normalize(torch.randn(self.N, cfg.phase_dim), dim=-1))

    def encode_question(self, questions: Tensor) -> Tensor:
        if self.question_encoder is not None:
            return self.question_encoder(questions)
        if self.q_vocab:
            return F.one_hot(questions.long(), self.q_vocab).float().flatten(-2)
        return questions.float()

    def forward(self, batch: dict, t_override: int | None = None) -> dict:
        q = self.encode_question(batch['questions'])
        B = q.shape[0]
        f = self.pathways.encoder_film(self.field_enc(batch['images']), q, self.pos_emb)
        Zt = self.field.to_tokens(self.field(f))
        X, anchors = self.binder(f.flatten(2).transpose(1, 2), Zt)
        z_slots = F.normalize(self.anchor_to_phase(anchors.flatten(2)), dim=-1)
        X = self.pathways.content_film(X, q)

        h_slots, h_head = self.pathways.init_states(q)
        h = torch.cat([h_slots + self.identity.embeds.unsqueeze(0), h_head], 1)
        X = torch.cat([X, X.new_zeros(B, 1, X.shape[-1])], 1)
        z = F.normalize(torch.randn(B, self.N, self.d, device=X.device, dtype=X.dtype), dim=-1)
        z = torch.cat([z_slots, z[:, self.M:]], 1)
        if self.cfg.addresses == 'static':
            z = F.normalize(self.z_static, dim=-1).expand(B, -1, -1)

        for _ in range(self.T if t_override is None else int(t_override)):
            r = self.medium(h, z)
            h = self.identity.step(torch.cat([X, r], -1), h)
            if self.cfg.addresses == 'computed':
                z = self.dynamics(z, h)
        return {'logits': self.readout(h[:, self.M], q)}

    @classmethod
    def from_config(cls, cfg: SyncNetConfig, dataset: str, answer_dim: int) -> 'SyncNet':
        return cls(cfg, dataset, answer_dim)
