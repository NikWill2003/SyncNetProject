"""The canonical SyncNet in one file, with no branches.

Everything the thesis reports is this model. The only knobs are sizes and
the two per-task perception settings (stem, bias initialisation); every
ablation of the thesis is a *different* model and lives in the main
codebase, not here.

    SyncNet(task='sort_of_clevr')                       # field trunk, learned bias
    SyncNet(task='sqoop')                               # shared cnn stem, partition-seeded bias
    logits = model(images, questions)                   # images (B,3,S,S), questions (B,q)

Parameter names match src/models/sync/identity_busnet.py exactly, so a
checkpoint trained there loads here with load_state_dict(strict=True).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---- fixed sizes of the oscillator field (Chapter 3, Table 3.1) ----
FIELD_CH = 64          # feature channels the field reads
FIELD_HIDDEN = 48      # width of the field trunk
FIELD_GROUPS = 16      # oscillators per cell
FIELD_OSC_D = 4        # dimension of each cell oscillator
T_FIELD = 8            # field steps T_x
DT_FIELD = 1.0         # field step size gamma_x
TOK_DIM = 64           # slot content width
BETA = 8.0             # competition inverse temperature (initial; learned)

TASKS = {
    'sort_of_clevr': dict(img_size=75, q_size=18, q_vocab=None, answer_dim=10, stem='field', bias_init='zero'),
    'sqoop': dict(img_size=64, q_size=3, q_vocab=40, answer_dim=2, stem='cnn', bias_init='partition'),
}


# =============================================================================
# Perception: stem -> oscillator field -> seeded competition
# =============================================================================
class FieldEncoder(nn.Module):
    """The Sort-of-CLEVR stem: two stride-2 stages then a 3x3 head -> (B, 64, S, S)."""

    def __init__(self, img_size: int) -> None:
        super().__init__()
        layers, cin, s = [], 3, img_size
        for _ in range(2):
            layers += [nn.Conv2d(cin, FIELD_HIDDEN, 3, stride=2, padding=1), nn.GroupNorm(8, FIELD_HIDDEN), nn.SiLU()]
            cin, s = FIELD_HIDDEN, (s + 1) // 2
        layers += [nn.Conv2d(cin, FIELD_HIDDEN, 3, padding=1), nn.GroupNorm(8, FIELD_HIDDEN), nn.SiLU(),
                   nn.Conv2d(FIELD_HIDDEN, FIELD_CH, 3, padding=1)]
        self.net = nn.Sequential(*layers)
        self.spatial = s

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SQOOPStem(nn.Module):
    """The stem every SQOOP model shares: six 3x3 convolutions, two max-pools -> (B, ch, S/4, S/4)."""

    def __init__(self, img_size: int, ch: int = 64) -> None:
        super().__init__()
        block = lambda cin: [nn.Conv2d(cin, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU()]
        self.cnn = nn.Sequential(*block(3), *block(ch), nn.MaxPool2d(2), *block(ch), *block(ch), nn.MaxPool2d(2), *block(ch), *block(ch))
        self.spatial = img_size // 4
        self.ch = ch

    def forward(self, x: Tensor) -> Tensor:
        return self.cnn(x)


class AdaptedTrunk(nn.Module):
    """A shared stem adapted to the field interface by a 1x1 conv to FIELD_CH."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.adapt = nn.Conv2d(base.ch, FIELD_CH, 1)
        self.spatial = base.spatial

    def forward(self, x: Tensor) -> Tensor:
        return self.adapt(self.base(x))


class OscillatorField(nn.Module):
    """Equation (field): FIELD_GROUPS unit vectors per cell, T_FIELD Euler steps of
    natural rotation + 5x5 conv coupling + feature stimulus, tangent-projected, renormalised."""

    def __init__(self) -> None:
        super().__init__()
        self.d, self.K, self.C = FIELD_OSC_D, FIELD_GROUPS, FIELD_OSC_D * FIELD_GROUPS
        self.T, self.dt = T_FIELD, DT_FIELD
        self.z_head = nn.Conv2d(FIELD_CH, self.C, 1)
        self.stim = nn.Conv2d(FIELD_CH, self.C, 1)
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
        return z.view(B, self.K, self.d, S * S).permute(0, 3, 1, 2)          # (B, P, K, d)

    def forward(self, f: Tensor) -> Tensor:
        z = self.normalise(self.z_head(f))
        c = self.stim(f)
        for _ in range(self.T):
            z = self.normalise(z + self.dt * (self.rotate(z) + self.tangent(z, self.J(z) + c)))
        return z


class CompetitiveClaim(nn.Module):
    """Equations (anchor) and (competition): per-module anchor priors, then three
    rounds of softmax-over-slots assignment seeded by the spatial bias b_k(p)."""

    def __init__(self, n_slots: int, n_cells: int, bias_init: str, bias_scale: float, iters: int = 3) -> None:
        super().__init__()
        self.n_slots, self.iters = n_slots, iters
        bias = torch.zeros(n_slots, n_cells)
        if bias_init == 'partition':                                         # one grid region per slot
            S = int(round(n_cells ** 0.5))
            rows = max(1, int(n_slots ** 0.5)); cols = -(-n_slots // rows)
            ys = torch.arange(S).repeat_interleave(S); xs = torch.arange(S).repeat(S)
            region = (ys * rows // S) * cols + (xs * cols // S)
            for k in range(min(n_slots, rows * cols)):
                bias[k, region == k] = bias_scale
        self.cell_bias = nn.Parameter(bias)
        self.register_buffer('prior_scale', torch.ones(()))
        self.anchor_mu = nn.Parameter(torch.randn(1, n_slots, FIELD_GROUPS, FIELD_OSC_D))
        self.anchor_log_sigma = nn.Parameter(torch.zeros(1, n_slots, FIELD_GROUPS, FIELD_OSC_D))
        self.log_beta = nn.Parameter(torch.log(torch.tensor(BETA)))
        self.to_slot = nn.Sequential(nn.LayerNorm(FIELD_CH), nn.Linear(FIELD_CH, TOK_DIM))

    def forward(self, feats: Tensor, Zt: Tensor) -> tuple[Tensor, Tensor]:
        """feats (B, P, FIELD_CH); Zt (B, P, K, d) -> slot contents (B, M, TOK_DIM), anchors (B, M, K, d)."""
        B, K_f = feats.shape[0], Zt.shape[2]
        eps = torch.randn(B, self.n_slots, FIELD_GROUPS, FIELD_OSC_D, device=feats.device, dtype=feats.dtype)
        phi = F.normalize(self.anchor_mu + self.anchor_log_sigma.exp() * eps, dim=-1)
        for _ in range(self.iters):
            logits = self.log_beta.exp() * torch.einsum('bkgd,bngd->bkn', phi, Zt) / K_f
            logits = logits + self.prior_scale * self.cell_bias.unsqueeze(0)
            attn = F.softmax(logits, dim=1)                                  # cells choose slots
            reads = attn / (attn.sum(-1, keepdim=True) + 1e-8)
            phi = F.normalize(torch.einsum('bkn,bngd->bkgd', reads, Zt), dim=-1)
            slots = self.to_slot(torch.einsum('bkn,bnf->bkf', reads, feats))
        return slots, phi


# =============================================================================
# The question's four pathways, in one place
# =============================================================================
class QuestionPathways(nn.Module):
    def __init__(self, q_size: int, n_modules: int, module_dim: int) -> None:
        super().__init__()
        self.grid_film_gamma = nn.Linear(q_size, FIELD_CH)
        self.grid_film_beta = nn.Linear(q_size, FIELD_CH)
        self.grid_norm = nn.GroupNorm(8, FIELD_CH, affine=True)
        self.film_gamma = nn.Linear(q_size, TOK_DIM)
        self.film_beta = nn.Linear(q_size, TOK_DIM)
        self.norm = nn.LayerNorm(TOK_DIM)
        self.h_init = nn.Sequential(nn.Linear(q_size, 64), nn.GELU(), nn.Linear(64, n_modules * module_dim))
        self.head_init = nn.Sequential(nn.Linear(q_size, 64), nn.GELU(), nn.Linear(64, module_dim))
        self.head_embed = nn.Parameter(torch.randn(1, module_dim) / module_dim ** 0.5)
        self.n_modules, self.module_dim = n_modules, module_dim

    def encoder_film(self, f: Tensor, q: Tensor, pos_emb: Tensor) -> Tensor:
        f = f * (1 + self.grid_film_gamma(q))[..., None, None] + self.grid_film_beta(q)[..., None, None]
        return self.grid_norm(f) + pos_emb

    def content_film(self, X: Tensor, q: Tensor) -> Tensor:
        return self.norm(X * (1 + self.film_gamma(q)).unsqueeze(1) + self.film_beta(q).unsqueeze(1))

    def init_states(self, q: Tensor) -> tuple[Tensor, Tensor]:
        h_slots = self.h_init(q).reshape(q.shape[0], self.n_modules, self.module_dim)
        h_head = self.head_init(q).unsqueeze(1) + self.head_embed.unsqueeze(0)
        return h_slots, h_head


# =============================================================================
# Modules, medium, phase dynamics, readout
# =============================================================================
class PrivateCells(nn.Module):
    """Equation (update): one GRU per module (slots and head), plus the identity embeddings."""

    def __init__(self, n_rows: int, in_dim: int, module_dim: int, n_modules: int) -> None:
        super().__init__()
        self.cells = nn.ModuleList(nn.GRUCell(in_dim, module_dim) for _ in range(n_rows))
        self.embeds = nn.Parameter(torch.randn(n_modules, module_dim) / module_dim ** 0.5)

    def step(self, inp: Tensor, h: Tensor) -> Tensor:
        return torch.stack([cell(inp[:, k], h[:, k]) for k, cell in enumerate(self.cells)], 1)

    def embed(self, h_slots: Tensor) -> Tensor:
        return h_slots + self.embeds.unsqueeze(0)


class SharedBus(nn.Module):
    """Equation (bus): B = sum_j m_j z_j^T; row k reads B through its frame Q_k, echo cancelled, / N."""

    def __init__(self, module_dim: int, msg_dim: int, phase_dim: int, ref_seed: int = 1234) -> None:
        super().__init__()
        self.msg_dim, self.phase_dim = msg_dim, phase_dim
        self.msg_proj = nn.Linear(module_dim, msg_dim)
        g = torch.Generator().manual_seed(ref_seed)
        self.register_buffer('frame_ref', F.normalize(torch.randn(phase_dim - 1, phase_dim, generator=g), dim=-1))

    @property
    def out_dim(self) -> int:
        return self.msg_dim * self.phase_dim

    def _frame(self, z: Tensor) -> Tensor:
        """(B, N, d) -> (B, N, d, d): first vector z_k, the rest Gram-Schmidt of fixed references."""
        vecs = [z]
        for k in range(self.phase_dim - 1):
            v = self.frame_ref[k].to(z.dtype).expand_as(z)
            for u in vecs:
                v = v - (v * u).sum(-1, keepdim=True) * u
            vecs.append(F.normalize(v, dim=-1))
        return torch.stack(vecs, 2)

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        N = h.shape[1]
        m = self.msg_proj(h)                                                 # (B, N, d_m)
        bus = torch.einsum('bnD,bnd->bDd', m, z)                             # one wire
        Fr = self._frame(z)
        r = torch.einsum('bDd,bnad->bnDa', bus, Fr) - torch.einsum('bnD,bnd,bnad->bnDa', m, z, Fr)
        return r.flatten(2) / float(N)


class KuramotoStep(nn.Module):
    """Equation (phi): z <- Pi( z + dt ( omega A z + Proj_z( (J * kappa(h)) z + W_c h ) ) ),
    with kappa_kj = tanh(MLP([h_k; h_j])) and A a learned antisymmetric generator."""

    def __init__(self, n_rows: int, module_dim: int, phase_dim: int, dt: float, k_hidden: int = 64) -> None:
        super().__init__()
        self.dt = dt
        self.omega = nn.Parameter(torch.zeros(n_rows))
        self.K = nn.Parameter(torch.ones(n_rows, n_rows))
        self.k_mlp = nn.Sequential(nn.Linear(2 * module_dim, k_hidden), nn.GELU(), nn.Linear(k_hidden, 1), nn.Tanh())
        self.stim = nn.Linear(module_dim, phase_dim)
        self.gen = nn.Module()
        self.gen.raw = nn.Parameter(0.1 * torch.randn(phase_dim, phase_dim))

    @staticmethod
    def tangent(z: Tensor, v: Tensor) -> Tensor:
        return v - (v * z).sum(-1, keepdim=True) * z

    def forward(self, z: Tensor, h: Tensor) -> Tensor:
        B, N, dm = h.shape
        A = self.gen.raw - self.gen.raw.t()
        A = A / (A.norm() / math.sqrt(2) + 1e-6)
        vel = self.omega.to(z.dtype)[None, :, None] * torch.einsum('de,bne->bnd', A, z)
        hi, hj = h.unsqueeze(2).expand(B, N, N, dm), h.unsqueeze(1).expand(B, N, N, dm)
        kap = self.k_mlp(torch.cat([hi, hj], -1)).squeeze(-1)
        pull = torch.einsum('bij,bjd->bid', self.K.to(z.dtype).unsqueeze(0) * kap, z)
        vel = vel + self.tangent(z, pull) + self.tangent(z, self.stim(h))
        return F.normalize(z + self.dt * vel, dim=-1)


class HeadReadout(nn.Module):
    """Equation (readout): y = MLP([h_0; q]). No question-only term."""

    def __init__(self, module_dim: int, q_size: int, answer_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.head_out = nn.Sequential(nn.Linear(module_dim + q_size, hidden), nn.GELU(), nn.Linear(hidden, answer_dim))
        self.prior_head = None

    def forward(self, h_head: Tensor, q: Tensor) -> Tensor:
        return self.head_out(torch.cat([h_head, q], -1))


# =============================================================================
# The model
# =============================================================================
class SyncNet(nn.Module):
    def __init__(self, task: str, n_modules: int = 6, phase_dim: int = 6, t_bus: int = 8, dt: float = 0.5,
                 module_dim: int = 96, msg_dim: int = 4, bias_scale: float = 4.0, stem_ch: int = 64,
                 stem: str | None = None, bias_init: str | None = None) -> None:
        super().__init__()
        spec = TASKS[task]
        stem, bias_init = stem or spec['stem'], bias_init or spec['bias_init']
        self.q_vocab = spec['q_vocab']
        self.q_size = spec['q_size'] * (self.q_vocab or 1)
        self.M, self.d, self.T = n_modules, phase_dim, t_bus
        self.N = n_modules + 1                                               # slots + the head

        self.field_enc = FieldEncoder(spec['img_size']) if stem == 'field' else AdaptedTrunk(SQOOPStem(spec['img_size'], stem_ch))
        S = self.field_enc.spatial
        self.pos_emb = nn.Parameter(0.02 * torch.randn(1, FIELD_CH, S, S))
        self.field = OscillatorField()
        self.binder = CompetitiveClaim(n_modules, S * S, bias_init, bias_scale)
        self.anchor_to_phase = nn.Linear(FIELD_GROUPS * FIELD_OSC_D, phase_dim)
        self.pathways = QuestionPathways(self.q_size, n_modules, module_dim)
        self.medium = SharedBus(module_dim, msg_dim, phase_dim)
        self.identity = PrivateCells(self.N, TOK_DIM + self.medium.out_dim, module_dim, n_modules)
        self.dynamics = KuramotoStep(self.N, module_dim, phase_dim, dt)
        self.readout = HeadReadout(module_dim, self.q_size, spec['answer_dim'])

    def encode_question(self, questions: Tensor) -> Tensor:
        if self.q_vocab:
            return F.one_hot(questions.long(), self.q_vocab).float().flatten(-2)
        return questions.float()

    def forward(self, images: Tensor, questions: Tensor, t_override: int | None = None) -> Tensor:
        q = self.encode_question(questions)                                  # (B, q_size)
        B = q.shape[0]

        # -- perception: stem -> field -> seeded competition -> slot contents and starting phases
        f = self.pathways.encoder_film(self.field_enc(images), q, self.pos_emb)
        Zt = self.field.to_tokens(self.field(f))                             # (B, P, K, d_f)
        X, anchors = self.binder(f.flatten(2).transpose(1, 2), Zt)           # (B, M, TOK_DIM), (B, M, K, d_f)
        z_slots = F.normalize(self.anchor_to_phase(anchors.flatten(2)), dim=-1)
        X = self.pathways.content_film(X, q)

        # -- modules: states from what they won and the question; the head from the question alone
        h_slots, h_head = self.pathways.init_states(q)
        h = torch.cat([self.identity.embed(h_slots), h_head], 1)             # (B, N, d_h)
        X = torch.cat([X, torch.zeros(B, 1, X.shape[-1], device=X.device, dtype=X.dtype)], 1)
        z = F.normalize(torch.randn(B, self.N, self.d, device=X.device, dtype=X.dtype), dim=-1)   # drawn for every row so RNG use matches the reference
        z = torch.cat([z_slots, z[:, self.M:]], 1)                          # slots start from their anchors; the head from its draw

        # -- T steps of communicate / update / advance phase
        for _ in range(self.T if t_override is None else int(t_override)):
            r = self.medium(h, z)
            h = self.identity.step(torch.cat([X, r], -1), h)
            z = self.dynamics(z, h)

        return self.readout(h[:, self.M], q)                                 # from the head alone
