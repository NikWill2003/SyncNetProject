from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

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
    GateShape, HebbianCoupling, SkewGenerator, angle_to_unit, random_unit,
    sphere_step, straight_through_topk, tangent, zero_diag,
)

"""One SyncNet, composed from independent axes.

Every axis below defaults to the behaviour of the model that produced the
gate-null cells, so existing configs reproduce existing runs exactly. The
new axes exist to answer two questions the null could not: is the model
competent enough for the gate to matter, and can the gate express the
selectivity the thesis attributes to it.

Axes (defaults first):
  input.q_conditioning  film | broadcast_cat | token
  input.partition       none | quadrant | views | object
      object   : module k may read object k only (needs n_modules ==
                 n_objects and the objects encoder): every relation
                 must then pass through a message
  encoder               patchify | cnn | objects      (encoder: {name: ...})
      objects  : no perception -- six exact object descriptors read off
                 the pixels (object_tokens.py); the "no patchify" input
  read.n_read_heads     1 | H       H independent queries per module per step
  read.read_proj        false | true  fold the H reads back to ch before the
                                      GRU (keeps the cell size independent of H)
  read.read_norm        tokens | modules | both
      tokens  : softmax over tokens per module (attention)
      modules : softmax over modules per token, then mean per module
                (slot-attention competition: tokens choose a module)
  update.per_module_gru false | true  separate GRUCell per module
  gate.mode             phase | frozen | attn | mlp | open | zero | phase_io
      zero     : no communication at all (lower bound)
      phase_io : each module carries a receive phase and a send phase,
                 g_ij = shape(cos(theta_in_i - theta_out_j)); the directed
                 (communication-through-coherence) reading of the gate
  gate.sharpen          false | true  (1+cos)/2  vs  sigmoid(alpha cos + b)
  gate.zero_diag        false | true  drop the self-message
  gate.topk             0 | k         hard top-k senders (straight-through)
  osc_dim               2 | d         scalar phase | unit vector on S^{d-1}
  theta_init            random | learned | zero
  coupling              mlp | hebbian | none      kappa_ij(h) in eq. phi
  drive                 none | stimulus | rotate  an input term in eq. phi
      stimulus : AKOrN's conditional stimulus, c_i = W h_i pulls z_i
      rotate   : a per-module rotation predicted from h_i (the coalitions
                 'mlp' update: persistent angle, no interaction)
  msg_agg               mean | bus | budget
      budget   : divide by the open mass, so opening a channel dilutes
                 every other one (open is no longer free)
  readout.mode          concat | sync | both | sum

Runtime overrides (never in config; used by the interventions callback):
  gate_override   open | zero | frozen | shuffle
  phase_override  freeze | shuffle
  t_override      int
"""

Partition = Literal['none', 'quadrant', 'views']
QCond = Literal['film', 'broadcast_cat', 'token']
GateMode = Literal['phase', 'attn', 'mlp', 'open', 'frozen', 'zero', 'phase_io']
MsgAgg = Literal['mean', 'bus', 'budget']
ReadoutMode = Literal['concat', 'sync', 'both', 'sum']

N_VIEWS = 3

PHASE_GATES = ('phase', 'frozen', 'phase_io')


@dataclass
class VQASyncNetConfig(ModelConfig):
    q_conditioning: str = 'film' # film | broadcast_cat | token
    q_emb_dim: int = 32
    partition: str = 'none' # none | quadrant | views
    use_pos_emb: bool = True

    # modules
    n_modules: int = 4
    module_dim: int = 128
    content_dim: int = 8
    query_hidden: int = 64
    beta_init: float = 5.0
    learn_beta: bool = True
    use_module_embed: bool = False
    embed_scale: float = 1.0

    # read
    n_read_heads: int = 1
    read_norm: str = 'tokens'           # tokens | modules | both
    read_proj: bool = False             # project H*ch -> ch before the GRU

    # update
    per_module_gru: bool = False

    # communication
    gate_mode: str = 'phase' # phase | attn | mlp | open | frozen | zero | phase_io
    T: int = 6
    dt: float = 0.1
    omega_init: float = 0.5
    learn_omega: bool = True
    k_hidden: int = 64
    deterministic_phase: bool = False   # legacy alias for theta_init='zero'
    theta_init: str = 'random'          # random | learned | zero
    osc_dim: int = 2
    coupling: str = 'mlp'               # mlp | hebbian | none
    drive: str = 'none'                 # none | stimulus | rotate
    gate_sharpen: bool = False
    gate_alpha_init: float = 4.0
    gate_bias_init: float = -1.0
    learn_sharpen: bool = True
    gate_zero_diag: bool = False
    gate_topk: int = 0

    # messages
    msg_dim: int = 64
    msg_agg: str = 'mean' # mean | bus | budget

    # readout: concat | sync | both | sum
    readout_mode: str = 'concat'
    hidden_dim: int = 128
    use_prior_head: bool = True # sum only
    vote_sees_q: bool = True # sum only

    # encoder as a plain dict: {'name': 'patchify'|'cnn', 'ch': ..., ...}
    encoder: dict[str, Any] = field(default_factory=lambda: {
        'name': 'patchify', 'ch': 128, 'patch_size': 5,
    })


def _sobel() -> Tensor:
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    return torch.stack([kx, kx.t().contiguous()]).unsqueeze(1)


def _norm_entropy(p: Tensor) -> Tensor:
    """Entropy of distributions on the last dim, normalised to [0, 1]."""
    n = p.shape[-1]
    ent = -(p.clamp(min=1e-12) * p.clamp(min=1e-12).log()).sum(-1)
    return ent / float(np.log(n))


class VQASyncNet(nn.Module):

    has_rotors = False
    is_syncnet = True
    SUPPORTED_OVERRIDES = frozenset({'t_override', 'gate_override',
                                     'phase_override', 'return_trace'})
    GATE_OVERRIDES = frozenset({'open', 'zero', 'frozen', 'shuffle'})
    PHASE_OVERRIDES = frozenset({'freeze', 'shuffle'})

    def __init__(
            self,
            cfg: VQASyncNetConfig,
            q_encoder: QuestionEncoder,
            img_size: int,
            answer_dim: int,
            object_colours: list[tuple[int, int, int]] | None = None,
            ) -> None:

        super().__init__()
        self.cfg = cfg
        self.q_encoder = q_encoder
        q_dim = q_encoder.out_dim
        enc = dict(cfg.encoder)
        enc_name = enc.get('name', 'patchify')
        if enc_name not in ('patchify', 'cnn', 'objects'):
            raise ValueError(f'unsupported encoder for syncnet: {enc_name!r}')
        self.q_dim, self.answer_dim = q_dim, answer_dim
        self.objects = enc_name == 'objects'

        if cfg.partition == 'views' and cfg.n_modules != N_VIEWS:
            raise ValueError(f'views partition requires n_modules={N_VIEWS}')
        if cfg.partition == 'quadrant' and cfg.n_modules != 4:
            raise ValueError('quadrant partition requires n_modules=4')
        if self.objects and cfg.partition in ('quadrant', 'views'):
            raise ValueError(f'{cfg.partition!r} partition needs a grid encoder')
        if cfg.partition == 'object' and not self.objects:
            raise ValueError("partition='object' requires the objects encoder")
        if self.objects and object_colours is None:
            raise ValueError('objects encoder needs the task colour table')
        if cfg.gate_mode == 'phase_io' and cfg.osc_dim != 2:
            raise ValueError('phase_io is implemented for osc_dim=2 only')
        if cfg.osc_dim < 2:
            raise ValueError('osc_dim must be >= 2')

        M, d = cfg.n_modules, cfg.module_dim
        self.M, self.d = M, d
        self.T, self.dt = cfg.T, cfg.dt
        H = max(1, int(cfg.n_read_heads))
        self.H = H
        self.theta_init = 'zero' if cfg.deterministic_phase else cfg.theta_init
        # scalar-angle path (exact legacy dynamics) or unit-vector path
        self.scalar_phase = cfg.osc_dim == 2

        # ---- input pathway ------------------------------------------
        enc_ch = int(enc['ch'])
        if cfg.q_conditioning == 'broadcast_cat':
            self.ch = enc_ch + cfg.q_emb_dim
            self.q_enc = nn.Linear(q_dim, cfg.q_emb_dim)
        else:
            self.ch = enc_ch
            if cfg.q_conditioning == 'film':
                self.film_gamma = nn.Linear(q_dim, enc_ch)
                self.film_beta = nn.Linear(q_dim, enc_ch)
            else:                                   # 'token'
                self.q_enc = nn.Linear(q_dim, enc_ch)

        n_enc = N_VIEWS if cfg.partition == 'views' else 1
        if self.objects:
            self.obj_tok = ObjectTokenizer(
                object_colours, img_size, int(enc.get('obj_size', 5)))  # type: ignore[arg-type]
            self.obj_embed = nn.Linear(self.obj_tok.feat_dim, enc_ch)
            self.encoders = nn.ModuleList([])
            self.spatial = None
            self.n_tokens = self.obj_tok.n_objects
            if cfg.partition == 'object' and cfg.n_modules != self.n_tokens:
                raise ValueError(
                    f'object partition requires n_modules={self.n_tokens}')
        else:
            self.encoders = nn.ModuleList([
                build_encoder(enc, img_size) for _ in range(n_enc)])
            self.spatial = self.encoders[0].spatial
            self.n_tokens = self.spatial * self.spatial
        self.norms = nn.ModuleList([
            nn.GroupNorm(8, enc_ch, affine=True) for _ in range(n_enc)])
        if cfg.use_pos_emb and not self.objects:
            self.pos_emb = nn.Parameter(
                0.02 * torch.randn(n_enc, enc_ch, self.spatial, self.spatial))
        if cfg.partition == 'object':
            self.register_buffer(
                'partition_mask', torch.eye(cfg.n_modules, dtype=torch.bool))

        if cfg.partition == 'views':
            self.register_buffer('sobel', _sobel())
            ys, xs = torch.meshgrid(
                torch.linspace(0, 1, img_size),
                torch.linspace(0, 1, img_size), indexing='ij')
            self.register_buffer('coords', torch.stack([xs, ys])[None])
        elif cfg.partition == 'quadrant':
            S, half = self.spatial, (self.spatial + 1) // 2
            mask = torch.zeros(M, S, S, dtype=torch.bool)
            mask[0, :half, :half] = True; mask[1, :half, half:] = True
            mask[2, half:, :half] = True; mask[3, half:, half:] = True
            self.register_buffer('partition_mask', mask.reshape(M, -1))

        # ---- module state -------------------------------------------
        self.h_init = nn.Sequential(
            nn.Linear(q_dim, cfg.query_hidden), nn.GELU(),
            nn.Linear(cfg.query_hidden, M * d))
        if cfg.use_module_embed:
            self.module_embed = nn.Parameter(
                cfg.embed_scale * torch.randn(M, d) / (d ** 0.5))
            self.embed_to_query = nn.Linear(d, self.ch * H)
        self.attn_query = nn.Linear(d, self.ch * H)
        log_beta = torch.tensor(float(np.log(cfg.beta_init)))
        if cfg.learn_beta:
            self.log_beta = nn.Parameter(log_beta)
        else:
            self.register_buffer('log_beta', log_beta)

        # ---- messages + gates ---------------------------------------
        self.msg_proj = nn.Linear(d, cfg.msg_dim)
        if cfg.msg_agg == 'bus':
            self.bus_norm = nn.LayerNorm(cfg.msg_dim)
        self.read_proj = None
        if cfg.read_proj and H > 1:
            self.read_proj = nn.Linear(self.ch * H, self.ch)
        a_dim = self.ch if self.read_proj is not None else self.ch * H
        cell_in = a_dim + cfg.msg_dim
        if cfg.per_module_gru:
            self.cells = nn.ModuleList(
                [nn.GRUCell(cell_in, d) for _ in range(M)])
        else:
            self.cell = nn.GRUCell(cell_in, d)

        omega0 = cfg.omega_init * torch.linspace(-1.0, 1.0, M)
        if cfg.learn_omega:
            self.omega = nn.Parameter(omega0)
        else:
            self.register_buffer('omega', omega0)
        self.K = nn.Parameter(torch.ones(M, M))
        if cfg.coupling == 'mlp':
            self.k_mlp = nn.Sequential(
                nn.Linear(2 * d, cfg.k_hidden), nn.GELU(),
                nn.Linear(cfg.k_hidden, 1), nn.Tanh())
        elif cfg.coupling == 'hebbian':
            self.k_hebb = HebbianCoupling(d, key_dim=32)
        elif cfg.coupling != 'none':
            raise ValueError(f'unknown coupling {cfg.coupling!r}')
        if cfg.gate_mode == 'mlp':
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * d, cfg.k_hidden), nn.GELU(),
                nn.Linear(cfg.k_hidden, 1), nn.Sigmoid())
        if cfg.gate_mode == 'attn':
            self.gate_q = nn.Linear(d, cfg.k_hidden)
            self.gate_k = nn.Linear(d, cfg.k_hidden)

        # --- new phase machinery (created only when switched on, so the
        # default parameter-creation order is unchanged) ---
        self.gate_shape = GateShape(
            cfg.gate_sharpen, cfg.gate_alpha_init, cfg.gate_bias_init,
            cfg.learn_sharpen)
        osc = cfg.osc_dim
        if self.theta_init == 'learned':
            if self.scalar_phase:
                self.theta0 = nn.Parameter(2 * np.pi * torch.rand(M))
            else:
                self.z0 = nn.Parameter(torch.randn(M, osc))
        if not self.scalar_phase:
            self.gen = SkewGenerator(osc, learn=True)
        if cfg.drive == 'stimulus':
            self.stim = nn.Linear(d, osc)
        elif cfg.drive == 'rotate':
            self.rot = nn.Sequential(
                nn.Linear(d, cfg.k_hidden), nn.GELU(),
                nn.Linear(cfg.k_hidden, 1 if self.scalar_phase else osc))
        elif cfg.drive != 'none':
            raise ValueError(f'unknown drive {cfg.drive!r}')
        if cfg.gate_mode == 'phase_io':
            self.omega_out = nn.Parameter(omega0.clone())
            self.K_out = nn.Parameter(torch.ones(M, M))
            if self.theta_init == 'learned':
                self.theta0_out = nn.Parameter(2 * np.pi * torch.rand(M))

        # ---- readout -------------------------------------------------
        self.n_pairs = M * (M - 1) // 2
        iu = torch.triu_indices(M, M, offset=1)
        self.register_buffer('pair_i', iu[0])
        self.register_buffer('pair_j', iu[1])
        if cfg.readout_mode in ('sync', 'both'):
            self.sync_decay_theta = nn.Parameter(-2.0 * torch.ones(self.n_pairs))
            self.sync_decay_h = nn.Parameter(-2.0 * torch.ones(self.n_pairs))

        self.content_head = nn.Linear(self.ch, cfg.content_dim)
        cdim = cfg.content_dim * H
        if cfg.readout_mode == 'sum':
            vote_in = cdim + d + (q_dim if cfg.vote_sees_q else 0)
            self.vote_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(vote_in, cfg.hidden_dim), nn.GELU(),
                              nn.Linear(cfg.hidden_dim, answer_dim))
                for _ in range(M)])
            if cfg.use_prior_head:
                self.prior_head = nn.Sequential(
                    nn.Linear(q_dim, cfg.hidden_dim), nn.GELU(),
                    nn.Linear(cfg.hidden_dim, answer_dim))
        else:
            sync_dim = 2 * self.n_pairs
            head_in = {'concat': M * cdim, 'sync': sync_dim,
                       'both': M * cdim + sync_dim}[cfg.readout_mode] + q_dim
            self.head = nn.Sequential(
                nn.Linear(head_in, cfg.hidden_dim), nn.GELU(),
                nn.Linear(cfg.hidden_dim, answer_dim))

    # ------------------------------------------------------------------
    # input pathway

    def _views(self, images: Tensor) -> list[Tensor]:
        gray = images.mean(dim=1, keepdim=True)
        edges = F.conv2d(gray, self.sobel, padding=1).abs()
        pos = torch.cat(
            [self.coords.expand(images.shape[0], -1, -1, -1) * gray, gray], 1)
        return [images, torch.cat([gray, edges], 1), pos]

    def _tokens(self, images: Tensor, q: Tensor):
        """-> tokens (B, M, P, ch) or (B, 1, P, ch), mask (M, P) or None."""
        cfg = self.cfg
        if self.objects:
            feats, found = self.obj_tok(images)
            self._last_found = found
            c = self.obj_embed(feats)                              # (B, P, ch)
            if cfg.q_conditioning == 'film':
                c = c * (1 + self.film_gamma(q)).unsqueeze(1) \
                    + self.film_beta(q).unsqueeze(1)
            c = self.norms[0](c.transpose(1, 2).unsqueeze(-1)
                              ).squeeze(-1).transpose(1, 2)
            if cfg.q_conditioning == 'broadcast_cat':
                qe = self.q_enc(q).unsqueeze(1).expand(-1, c.shape[1], -1)
                c = torch.cat([c, qe], dim=-1)
            return c.unsqueeze(1), getattr(self, 'partition_mask', None)
        srcs = self._views(images) if cfg.partition == 'views' else [images]
        outs = []
        for v, src in enumerate(srcs):
            c = self.encoders[v](src)
            if cfg.q_conditioning == 'film':
                c = c * (1 + self.film_gamma(q)[..., None, None]) \
                    + self.film_beta(q)[..., None, None]
            c = self.norms[v](c)
            if cfg.use_pos_emb:
                c = c + self.pos_emb[v].unsqueeze(0)
            t = c.permute(0, 2, 3, 1).reshape(c.shape[0], -1, c.shape[1])
            if cfg.q_conditioning == 'broadcast_cat':
                qe = self.q_enc(q).unsqueeze(1).expand(-1, t.shape[1], -1)
                t = torch.cat([t, qe], dim=-1)
            outs.append(t)
        tokens = torch.stack(outs, dim=1)                  # (B, V, P, ch)
        mask = getattr(self, 'partition_mask', None)
        return tokens, mask

    # ------------------------------------------------------------------
    # phase state: a dict so scalar / vector / io variants share one loop

    def _init_phase(self, B: int, dev) -> dict[str, Tensor]:
        M, osc = self.M, self.cfg.osc_dim
        if self.scalar_phase:
            if self.theta_init == 'zero':
                th = torch.zeros(B, M, device=dev)
            elif self.theta_init == 'learned':
                th = self.theta0.unsqueeze(0).expand(B, M)
            else:
                th = 2 * torch.pi * torch.rand(B, M, device=dev)
            st = {'theta': th}
            if self.cfg.gate_mode == 'phase_io':
                if self.theta_init == 'zero':
                    st['theta_out'] = torch.zeros(B, M, device=dev)
                elif self.theta_init == 'learned':
                    st['theta_out'] = self.theta0_out.unsqueeze(0).expand(B, M)
                else:
                    st['theta_out'] = 2 * torch.pi * torch.rand(B, M, device=dev)
            return st
        if self.theta_init == 'zero':
            z = torch.zeros(B, M, osc, device=dev); z[..., 0] = 1.0
        elif self.theta_init == 'learned':
            z = F.normalize(self.z0, dim=-1).unsqueeze(0).expand(B, M, osc)
        else:
            z = random_unit(B, M, osc, device=dev)
        return {'z': z}

    def _phase_unit(self, st: dict[str, Tensor]) -> Tensor:
        """(B, M, d) unit vectors for metrics / sync features."""
        return angle_to_unit(st['theta']) if self.scalar_phase else st['z']

    def _kappa(self, h: Tensor) -> Tensor | None:
        cfg = self.cfg
        if cfg.coupling == 'mlp':
            B, M, d = h.shape
            hi = h.unsqueeze(2).expand(B, M, M, d)
            hj = h.unsqueeze(1).expand(B, M, M, d)
            return self.k_mlp(torch.cat([hi, hj], -1)).squeeze(-1)
        if cfg.coupling == 'hebbian':
            return self.k_hebb(h)
        return None

    def _phase_step(self, st: dict[str, Tensor], h: Tensor) -> dict[str, Tensor]:
        cfg, dt = self.cfg, self.dt
        kap = self._kappa(h)
        if self.scalar_phase:
            theta = st['theta']
            if cfg.gate_mode == 'phase_io':
                th_in, th_out = theta, st['theta_out']
                # receive phases lock to the send phases they listen to;
                # send phases lock to the receive phases they reach
                d_in = th_out.unsqueeze(1) - th_in.unsqueeze(2)   # [b,i,j] out_j - in_i
                w = self.K * (kap if kap is not None else 1.0)
                v_in = self.omega + (w * torch.sin(d_in)).sum(-1)
                w_out = self.K_out * (kap.transpose(1, 2) if kap is not None else 1.0)
                v_out = self.omega_out + (w_out * torch.sin(-d_in.transpose(1, 2))).sum(-1)
                v_in = v_in + self._drive_scalar(th_in, h)
                v_out = v_out + self._drive_scalar(th_out, h)
                return {'theta': th_in + dt * v_in, 'theta_out': th_out + dt * v_out}
            vel = self.omega.expand_as(theta)
            if cfg.coupling != 'none':
                diff = theta.unsqueeze(1) - theta.unsqueeze(2)   # [b,i,j] theta_j - theta_i
                vel = vel + (self.K * torch.sin(diff) * kap).sum(-1)
            vel = vel + self._drive_scalar(theta, h)
            return {'theta': theta + dt * vel}
        # unit-vector path on S^{d-1}
        z = st['z']
        vel = self.gen(z, self.omega.expand(z.shape[:2]))
        if cfg.coupling != 'none':
            w = self.K.unsqueeze(0) * kap                        # (B, M, M)
            pull = torch.einsum('bij,bjd->bid', w, z)
            vel = vel + tangent(z, pull)
        vel = vel + self._drive_vector(z, h)
        return {'z': sphere_step(z, vel, dt)}

    def _drive_scalar(self, theta: Tensor, h: Tensor) -> Tensor:
        cfg = self.cfg
        if cfg.drive == 'stimulus':
            c = self.stim(h)                                    # (B, M, 2)
            # tangential component of c at theta = |c| sin(phi_c - theta)
            return -c[..., 0] * torch.sin(theta) + c[..., 1] * torch.cos(theta)
        if cfg.drive == 'rotate':
            return np.pi * torch.tanh(self.rot(h).squeeze(-1))
        return torch.zeros_like(theta)

    def _drive_vector(self, z: Tensor, h: Tensor) -> Tensor:
        cfg = self.cfg
        if cfg.drive == 'stimulus':
            return tangent(z, self.stim(h))
        if cfg.drive == 'rotate':
            return tangent(z, self.rot(h))
        return torch.zeros_like(z)

    # ------------------------------------------------------------------
    # gates

    def _phase_gate(self, st: dict[str, Tensor]) -> Tensor:
        if self.scalar_phase:
            if self.cfg.gate_mode == 'phase_io':
                dots = torch.cos(st['theta'].unsqueeze(-1) - st['theta_out'].unsqueeze(-2))
            else:
                th = st['theta']
                dots = torch.cos(th.unsqueeze(-1) - th.unsqueeze(-2))
        else:
            z = st['z']
            dots = torch.einsum('bid,bjd->bij', z, z)
        return self.gate_shape(dots)

    def _gates(self, st: dict[str, Tensor], h: Tensor) -> Tensor:
        cfg = self.cfg
        mode = cfg.gate_mode
        B, M, d = h.shape
        if mode in PHASE_GATES:
            g = self._phase_gate(st)
        elif mode == 'open':
            g = torch.ones(B, M, M, device=h.device)
        elif mode == 'zero':
            g = torch.zeros(B, M, M, device=h.device)
        elif mode == 'attn':
            gq, gk = self.gate_q(h), self.gate_k(h)
            g = F.softmax(torch.einsum('bik,bjk->bij', gq, gk)
                          / (gq.shape[-1] ** 0.5), dim=-1)
        else:
            hi = h.unsqueeze(2).expand(B, M, M, d)
            hj = h.unsqueeze(1).expand(B, M, M, d)
            g = self.gate_mlp(torch.cat([hi, hj], -1)).squeeze(-1)
        if cfg.gate_zero_diag:
            g = zero_diag(g)
        if cfg.gate_topk > 0:
            g = straight_through_topk(g, cfg.gate_topk, exclude_self=True)
        return g

    def _msg_norm(self) -> float:
        cfg = self.cfg
        if cfg.gate_topk > 0:
            return 1.0
        return 1.0 if cfg.gate_mode == 'attn' else float(self.M)

    # ------------------------------------------------------------------
    # read

    def _read(self, h: Tensor, flat: Tensor, pm: Tensor | None,
              per_module: bool, beta: Tensor):
        """-> attended content (B, M, H*ch), attention (B, M, H, P)."""
        cfg = self.cfg
        B, M, _ = h.shape
        H = self.H
        qv = self.attn_query(h)
        if cfg.use_module_embed:
            qv = qv + self.embed_to_query(self.module_embed).unsqueeze(0)
        queries = F.normalize(qv.view(B, M, H, self.ch), dim=-1)
        keys = F.normalize(flat, dim=-1)
        if per_module:                                         # (B, M, P, ch)
            logits = beta * torch.einsum('bmhc,bmpc->bmhp', queries, keys)
        else:                                                  # (B, P, ch)
            logits = beta * torch.einsum('bmhc,bpc->bmhp', queries, keys)
        if pm is not None:
            logits = logits.masked_fill(~pm[None, :, None, :], float('-inf'))
        if cfg.read_norm == 'tokens':
            attn = F.softmax(logits, dim=-1)
        elif cfg.read_norm == 'modules':
            w = F.softmax(logits, dim=1)                       # tokens choose modules
            attn = w / (w.sum(-1, keepdim=True) + 1e-6)
        elif cfg.read_norm == 'both':
            w = F.softmax(logits, dim=-1) * F.softmax(logits, dim=1)
            attn = w / (w.sum(-1, keepdim=True) + 1e-6)
        else:
            raise ValueError(f'unknown read_norm {cfg.read_norm!r}')
        if pm is not None:
            attn = attn.masked_fill(~pm[None, :, None, :], 0.0)
        a = (torch.einsum('bmhp,bmpc->bmhc', attn, flat) if per_module
             else torch.einsum('bmhp,bpc->bmhc', attn, flat))
        return a.reshape(B, M, H * self.ch), attn

    def _update(self, a: Tensor, msg: Tensor, h: Tensor) -> Tensor:
        B, M, d = h.shape
        inp = torch.cat([a, msg], -1)
        if self.cfg.per_module_gru:
            return torch.stack(
                [self.cells[k](inp[:, k], h[:, k]) for k in range(M)], dim=1)
        return self.cell(inp.reshape(B * M, -1), h.reshape(B * M, d)).reshape(B, M, d)

    def _sync_features(self, z_hist, h_hist, B) -> Tensor:
        if not z_hist:
            return torch.zeros(B, 2 * self.n_pairs, device=self.pair_i.device)
        z = torch.stack(z_hist, 1); h = torch.stack(h_hist, 1)   # (B, T, M, .)
        ages = torch.arange(z.shape[1] - 1, -1, -1,
                            device=z.device, dtype=z.dtype)

        def w(p):
            x = torch.exp(-F.softplus(p).unsqueeze(-1) * ages)
            return x / x.sum(-1, keepdim=True)

        dots = (z[:, :, self.pair_i] * z[:, :, self.pair_j]).sum(-1)
        st = torch.einsum('btp,pt->bp', dots, w(self.sync_decay_theta))
        hn = F.normalize(h, dim=-1)
        ch_ = (hn[:, :, self.pair_i] * hn[:, :, self.pair_j]).sum(-1)
        sh = torch.einsum('btp,pt->bp', ch_, w(self.sync_decay_h))
        return torch.cat([st, sh], -1)

    # ------------------------------------------------------------------

    def forward(self, images: Tensor, questions: Tensor,
                t_override: int | None = None, return_trace: bool = False,
                gate_override: str | None = None,
                phase_override: str | None = None,
                **batch) -> dict:
        cfg = self.cfg
        q = self.q_encoder.flat(questions)
        B, M, d = images.shape[0], self.M, self.d
        T = t_override if t_override is not None else self.T
        dev = images.device

        tokens, mask = self._tokens(images, q)             # (B, V, P, ch)
        per_module = tokens.shape[1] > 1                   # views mode
        flat = tokens if per_module else tokens[:, 0]      # (B,M,P,ch)|(B,P,ch)
        b_content = self.content_head(flat)
        P = tokens.shape[2]

        h = self.h_init(q).reshape(B, M, d)
        if cfg.use_module_embed:
            h = h + self.module_embed.unsqueeze(0)
        st = self._init_phase(B, dev)
        if phase_override == 'shuffle':
            perm = torch.argsort(torch.rand(B, M, device=dev), dim=1)
            st = {k: v.gather(1, perm if v.dim() == 2 else
                              perm.unsqueeze(-1).expand_as(v))
                  for k, v in st.items()}

        beta = self.log_beta.exp()
        pm = mask.to(dev) if mask is not None else None
        if pm is not None:
            attn = (pm.float() / pm.float().sum(-1, keepdim=True)
                    ).unsqueeze(0).expand(B, M, P)
        else:
            attn = torch.full((B, M, P), 1.0 / P, device=dev)
        attn_h = attn.unsqueeze(2).expand(B, M, self.H, P)    # (B, M, H, P) at T=0

        need_sync = cfg.readout_mode in ('sync', 'both')
        z_hist: list[Tensor] = []
        h_hist: list[Tensor] = []
        traces = ({'phase': [], 'gates': [], 'attn': [], 'h': []}
                  if return_trace else None)

        evolve_phase = (cfg.gate_mode in ('phase', 'phase_io')
                        and phase_override != 'freeze'
                        and gate_override != 'frozen')
        norm = self._msg_norm()
        g = self._gates(st, h)                             # T=0 default
        g_held = g if gate_override == 'frozen' else None
        g_hist: list[Tensor] = []
        for _ in range(T):
            if g_held is not None:
                g = g_held
            else:
                g = self._gates(st, h)
            if gate_override == 'open':
                g = torch.ones(B, M, M, device=dev); norm = float(M)
            elif gate_override == 'zero':
                g = torch.zeros(B, M, M, device=dev)
            elif gate_override == 'shuffle':
                g = g[torch.randperm(B, device=dev)]
            g_hist.append(g)

            m_all = self.msg_proj(h)
            if cfg.msg_agg == 'bus':
                msg = self.bus_norm(torch.einsum('bij,bjk->bik', g, m_all))
            elif cfg.msg_agg == 'budget':
                msg = torch.einsum('bij,bjk->bik', g, m_all) \
                    / (g.sum(-1, keepdim=True) + 1.0)
            else:
                msg = torch.einsum('bij,bjk->bik', g, m_all) / norm

            a, attn_h = self._read(h, flat, pm, per_module, beta)
            if self.read_proj is not None:
                a = self.read_proj(a)
            attn = attn_h.mean(2)                          # (B, M, P) for traces
            h = self._update(a, msg, h)
            if evolve_phase:
                st = self._phase_step(st, h)

            if need_sync:
                z_hist.append(self._phase_unit(st)); h_hist.append(h)
            if return_trace:
                traces['phase'].append(self._phase_unit(st).detach())   # type: ignore
                traces['gates'].append(g.detach())         # type: ignore
                traces['attn'].append(attn.detach())       # type: ignore
                traces['h'].append(h.detach())             # type: ignore

        m_content = (torch.einsum('bmhp,bmpd->bmhd', attn_h, b_content) if per_module
                     else torch.einsum('bmhp,bpd->bmhd', attn_h, b_content)
                     ).reshape(B, M, -1)

        if cfg.readout_mode == 'sum':
            parts = ([m_content[:, k], h[:, k]] + ([q] if cfg.vote_sees_q else [])
                     for k in range(M))
            module_logits = torch.stack(
                [self.vote_heads[k](torch.cat(p, -1))
                 for k, p in enumerate(parts)], dim=1)
            prior = (self.prior_head(q) if cfg.use_prior_head
                     else torch.zeros_like(module_logits[:, 0]))
            logits = prior + module_logits.sum(1)
            if return_trace:
                traces['module_logits'] = module_logits.detach()  # type: ignore
                traces['prior_logits'] = prior.detach()           # type: ignore
        else:
            pieces: list[Tensor] = []
            if cfg.readout_mode in ('concat', 'both'):
                pieces.append(m_content.flatten(1))
            if need_sync:
                pieces.append(self._sync_features(z_hist, h_hist, B))
            pieces.append(q)
            logits = self.head(torch.cat(pieces, -1))

        with torch.no_grad():
            zu = self._phase_unit(st)
            R = zu.mean(1).norm(dim=-1).mean().item()
            off = ~torch.eye(M, dtype=torch.bool, device=dev)
            metrics = {'phase_R': R, 'gate_offdiag': g[:, off].mean().item()}
            # selectivity of incoming channels (1 = every sender equal)
            gi = g / (g.sum(-1, keepdim=True) + 1e-6)
            metrics['gate_entropy'] = _norm_entropy(gi).mean().item()
            # segregation: overlap of what modules read (1 = same tokens)
            an = F.normalize(attn, dim=-1)
            ov = torch.einsum('bmp,bnp->bmn', an, an)
            metrics['read_overlap'] = ov[:, off].mean().item()
            metrics['read_entropy'] = _norm_entropy(attn).mean().item()
            # does the gate pattern change over the computation?
            if len(g_hist) > 1:
                gs = torch.stack(g_hist, 0)                    # (T, B, M, M)
                metrics['gate_tvar'] = gs.var(0, unbiased=False)[:, off].mean().item()
            if self.objects:
                metrics['obj_found'] = self._last_found.float().sum(-1).mean().item()
            if cfg.readout_mode == 'sum':
                v = F.normalize(module_logits, dim=-1)
                metrics['vote_agreement'] = torch.einsum(
                    'bmd,bnd->bmn', v, v)[:, off].mean().item()
                if cfg.use_prior_head:
                    metrics['evidence_ratio'] = (
                        module_logits.sum(1).norm(dim=-1).mean()
                        / (prior.norm(dim=-1).mean() + 1e-8)).item()

        return {'logits': logits, 'traces': traces, 'metrics': metrics}
