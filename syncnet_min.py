"""A minimal SyncNet: synchrony used once, between modules.

Kept (each has an ablation showing it matters): seeded competition over the
encoder's tokens, a phase per module initialised from its content and advanced
by a signed content-coupled Kuramoto step with a stimulus term, one shared wire
read through each module's orthonormal frame, a private GRU per module, and a
head that reads the bus and never sees the image.
Dropped: the oscillator field, per-module anchor priors, natural frequencies,
the pairwise coupling MLP, the question-only readout term.

Switches for the canonical-minus-one comparisons:
    medium in {bus, silent, attention}, addresses in {computed, static},
    slot_bias_init in {zero, partition, random}, competition_rounds (1 = no refinement).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.common.img_enc import build_image_encoder
from src.models.common.pos_enc import PositionalEncoder2D
from src.models.common.qst_enc import build_question_encoder


@dataclass
class SyncNetConfig:
    image_encoder: dict = field(default_factory=lambda: {'name': 'cnn'})
    question_encoder: dict = field(default_factory=lambda: {'name': 'identity'})

    n_modules: int = 6
    phase_dim: int = 6
    n_steps: int = 8
    phase_step_size: float = 0.5

    slot_dim: int = 64
    slot_key_dim: int = 64
    slot_bias_init: str = 'zero'       # zero | partition | random
    slot_bias_scale: float = 4.0
    competition_rounds: int = 3        # 1 = one softmax, no refinement of the slot queries

    module_dim: int = 96
    message_dim: int = 4
    kappa_dim: int = 64
    readout_hidden_dim: int = 128

    medium: str = 'bus'                # bus | silent | attention
    addresses: str = 'computed'        # computed | static


class SeededCompetition(nn.Module):
    """Tokens choose one slot each (softmax over slots), seeded by a spatial bias.
    Over `rounds`, each slot's query moves to the mean key of the tokens it won,
    so a slot can drift off its seed toward what it actually holds."""

    def __init__(self, n_slots: int, n_tokens: int, input_dim: int, slot_dim: int, key_dim: int, bias_init: str, bias_scale: float, rounds: int = 3) -> None:
        super().__init__()
        self.key_dim, self.rounds = key_dim, rounds
        self.slot_queries = nn.Parameter(torch.randn(n_slots, key_dim) / key_dim ** 0.5)
        self.to_key = nn.Linear(input_dim, key_dim, bias=False)
        self.to_slot = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, slot_dim))
        bias = torch.zeros(n_slots, n_tokens)
        if bias_init == 'partition':
            S = int(round(n_tokens ** 0.5))
            rows = max(1, int(n_slots ** 0.5)); cols = -(-n_slots // rows)
            ys = torch.arange(S).repeat_interleave(S); xs = torch.arange(S).repeat(S)
            region = (ys * rows // S) * cols + (xs * cols // S)
            for k in range(min(n_slots, rows * cols)):
                bias[k, region == k] = bias_scale
        elif bias_init == 'random':
            region = torch.randint(0, n_slots, (n_tokens,), generator=torch.Generator().manual_seed(0))
            for k in range(n_slots):
                bias[k, region == k] = bias_scale
        elif bias_init != 'zero':
            raise ValueError(f'unknown slot_bias_init {bias_init!r}')
        self.cell_bias = nn.Parameter(bias)

    def forward(self, tokens: Tensor) -> Tensor:
        keys = self.to_key(tokens)                                           # (B, P, key)
        queries = self.slot_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        for _ in range(self.rounds):
            logits = torch.einsum('bpd,bmd->bmp', keys, queries) / self.key_dim ** 0.5 + self.cell_bias
            reads = F.softmax(logits, dim=1)                                 # each token picks a slot
            reads = reads / (reads.sum(-1, keepdim=True) + 1e-8)
            queries = torch.einsum('bmp,bpd->bmd', reads, keys)             # query <- mean key of what it won
        return self.to_slot(torch.einsum('bmp,bpc->bmc', reads, tokens))    # (B, M, slot_dim)


class Frame:
    """Orthonormal frame per module: first axis its phase, the rest Gram-Schmidt of fixed references."""

    def __init__(self, phase_dim: int, seed: int = 1234) -> None:
        g = torch.Generator().manual_seed(seed)
        self.ref = F.normalize(torch.randn(phase_dim - 1, phase_dim, generator=g), dim=-1)

    def __call__(self, z: Tensor) -> Tensor:                                 # (B, N, d) -> (B, N, d, d)
        vecs = [z]
        for k in range(z.shape[-1] - 1):
            v = self.ref[k].to(z).expand_as(z)
            for u in vecs:
                v = v - (v * u).sum(-1, keepdim=True) * u
            vecs.append(F.normalize(v, dim=-1))
        return torch.stack(vecs, 2)


class Bus(nn.Module):
    """One wire: B = sum_j m_j z_j^T; module k reads B through its frame, own echo removed."""

    def __init__(self, module_dim: int, message_dim: int, phase_dim: int) -> None:
        super().__init__()
        self.msg = nn.Linear(module_dim, message_dim)
        self.frame = Frame(phase_dim)
        self.out_dim = message_dim * phase_dim

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        m = self.msg(h)
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
    """Per-sender access with a content gate: the content-routing alternative to the bus."""

    def __init__(self, module_dim: int, message_dim: int, phase_dim: int) -> None:
        super().__init__()
        self.msg = nn.Linear(module_dim, message_dim)
        self.q, self.k = nn.Linear(module_dim, module_dim), nn.Linear(module_dim, module_dim)
        self.out_dim = message_dim

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        N = h.shape[1]
        att = torch.einsum('bnd,bmd->bnm', self.q(h), self.k(h)) / h.shape[-1] ** 0.5
        att = att.masked_fill(torch.eye(N, dtype=torch.bool, device=h.device), float('-inf')).softmax(-1)
        return torch.einsum('bnm,bmD->bnD', att, self.msg(h))


class PhaseStep(nn.Module):
    """z_k <- Pi( z_k + gamma Proj_z( mean_j kappa_kj z_j + W_c h_k ) ),
    kappa_kj = tanh( a * cos(W h_k, W h_j) + b )  with a, b learned: signed, so
    modules can repel by content as well as attract (a=1, b=0 at init = cosine)."""

    def __init__(self, module_dim: int, phase_dim: int, kappa_dim: int, step: float) -> None:
        super().__init__()
        self.step = step
        self.proj = nn.Linear(module_dim, kappa_dim, bias=False)
        self.stim = nn.Linear(module_dim, phase_dim)
        self.gain = nn.Parameter(torch.ones(()))
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(self, z: Tensor, h: Tensor) -> Tensor:
        N = h.shape[1]
        c = F.normalize(self.proj(h), dim=-1)
        kappa = torch.tanh(self.gain * torch.einsum('bnc,bmc->bnm', c, c) + self.offset)
        kappa = kappa * (1 - torch.eye(N, device=h.device, dtype=h.dtype))
        drive = torch.einsum('bnm,bmd->bnd', kappa, z) / max(N - 1, 1) + self.stim(h)
        drive = drive - (drive * z).sum(-1, keepdim=True) * z
        return F.normalize(z + self.step * drive, dim=-1)


class SyncNet(nn.Module):
    MEDIA = {'bus': Bus, 'silent': Silent, 'attention': Attention}

    def __init__(self, cfg: SyncNetConfig, dataset: str, answer_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_encoder = build_image_encoder(cfg.image_encoder, dataset)
        self.question_encoder = build_question_encoder(cfg.question_encoder, dataset, allowed={'identity', 'mlp', 'lstm'})
        self.pos = PositionalEncoder2D(hidden_dim=self.image_encoder.ch, row_len=self.image_encoder.spatial, col_len=self.image_encoder.spatial)
        q_dim = self.question_encoder.output_shape[0]
        self.M, self.N, self.d, self.T = cfg.n_modules, cfg.n_modules + 1, cfg.phase_dim, cfg.n_steps

        self.binder = SeededCompetition(cfg.n_modules, self.image_encoder.n_tokens, self.image_encoder.ch, cfg.slot_dim, cfg.slot_key_dim,
                                        cfg.slot_bias_init, cfg.slot_bias_scale, cfg.competition_rounds)
        self.slot_init = nn.Sequential(nn.Linear(cfg.slot_dim + q_dim, cfg.module_dim), nn.GELU(), nn.Linear(cfg.module_dim, cfg.module_dim))
        self.head_init = nn.Sequential(nn.Linear(q_dim, cfg.module_dim), nn.GELU(), nn.Linear(cfg.module_dim, cfg.module_dim))
        self.slot_phase = nn.Linear(cfg.slot_dim, cfg.phase_dim)                 # a slot's address starts from its content
        self.head_phase = nn.Parameter(torch.randn(cfg.phase_dim))
        self.medium = self.MEDIA[cfg.medium](cfg.module_dim, cfg.message_dim, cfg.phase_dim)
        self.cells = nn.ModuleList(nn.GRUCell(cfg.slot_dim + self.medium.out_dim, cfg.module_dim) for _ in range(self.N))
        self.dynamics = PhaseStep(cfg.module_dim, cfg.phase_dim, cfg.kappa_dim, cfg.phase_step_size)
        if cfg.addresses == 'static':
            self.z_static = nn.Parameter(F.normalize(torch.randn(self.N, cfg.phase_dim), dim=-1))
        self.readout = nn.Sequential(nn.Linear(cfg.module_dim + q_dim, cfg.readout_hidden_dim), nn.GELU(), nn.Linear(cfg.readout_hidden_dim, answer_dim))

    def forward(self, batch: dict, t_override: int | None = None) -> dict:
        q = self.question_encoder(batch['questions'])
        tokens = self.pos(self.image_encoder(batch['images']).flatten(2).transpose(1, 2))
        slots = self.binder(tokens)                                          # (B, M, slot_dim)
        B = slots.shape[0]

        h = torch.cat([self.slot_init(torch.cat([slots, q.unsqueeze(1).expand(-1, self.M, -1)], -1)), self.head_init(q).unsqueeze(1)], 1)
        x = torch.cat([slots, slots.new_zeros(B, 1, slots.shape[-1])], 1)   # the head gets no image input
        if self.cfg.addresses == 'static':
            z = F.normalize(self.z_static, dim=-1).expand(B, -1, -1)
        else:
            z = F.normalize(torch.cat([self.slot_phase(slots), self.head_phase.expand(B, 1, -1)], 1), dim=-1)

        for _ in range(self.T if t_override is None else int(t_override)):
            r = self.medium(h, z)
            h = torch.stack([cell(torch.cat([x[:, k], r[:, k]], -1), h[:, k]) for k, cell in enumerate(self.cells)], 1)
            if self.cfg.addresses == 'computed':
                z = self.dynamics(z, h)
        return {'logits': self.readout(torch.cat([h[:, self.M], q], -1))}

    @classmethod
    def from_config(cls, cfg: SyncNetConfig, dataset: str, answer_dim: int) -> 'SyncNet':
        return cls(cfg, dataset, answer_dim)
