from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from omegaconf import MISSING
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ....core.config import ModelConfig
from ....core.encoders import EncoderConfig, PatchifyEncoder
from ..config import SortOfClevrDataConfig
from ..contracts import SortOfClevrOutput
from ..data import constants as C

"""One SyncNet, composed from independent axes.

Replaces ctm_syncnet / recurrent_syncnet_dev / partitioned_syncnet /
view_syncnet: each is now a preset over the same code, so an axis is
implemented (and fixed) exactly once.

    SortOfClevrSyncNetConfig()                       recurrent baseline
    SortOfClevrSyncNetConfig(readout_mode='sync')     ctm readout
    SortOfClevrSyncNetConfig(partition='quadrant',
                             readout_mode='sum')      partitioned
    SortOfClevrSyncNetConfig(partition='views',
                             n_modules=3,
                             msg_agg='bus')           semantic views

Axes:
  input.q_conditioning  film | broadcast_cat | token
      WHERE the question enters. The transformer sweep found this worth
      ~32 points of ternary accuracy (late token conditioning 0.547 vs
      per-token concatenation 0.865 at matched capacity), so it is an
      axis here, not a hardcoded choice.
  input.partition       none | quadrant | views
      Whether communication is NECESSARY (a module cannot see the whole
      scene) and whether relevance is sparse per sample.
  gate.mode             phase | attn | mlp | open | frozen
      What controls g_ij. 'phase' is the thesis mechanism (M-1 dof,
      persistent); 'attn' is the "why not just attention?" steelman.
  msg_agg               mean | bus
      Whether open gates are free (mean) or dilute every sender (bus).
  readout.mode          concat | sync | both | sum
      'sum' removes the head's covert cross-module integration channel:
      messages become the ONLY place information can combine.
"""

Partition = Literal['none', 'quadrant', 'views']
QCond = Literal['film', 'broadcast_cat', 'token']
GateMode = Literal['phase', 'attn', 'mlp', 'open', 'frozen']
MsgAgg = Literal['mean', 'bus']
ReadoutMode = Literal['concat', 'sync', 'both', 'sum']

N_VIEWS = 3


@dataclass
class SortOfClevrSyncNetConfig(ModelConfig):
    name: str = 'sort_of_clevr_syncnet'

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

    # communication
    gate_mode: str = 'phase' # phase | attn | mlp | open | frozen
    T: int = 6
    dt: float = 0.1
    omega_init: float = 0.5
    learn_omega: bool = True
    k_hidden: int = 64
    deterministic_phase: bool = False

    # messages
    msg_dim: int = 64
    msg_agg: str = 'mean' # mean | bus

    # readout: concat | sync | both | sum
    readout_mode: str = 'concat'
    hidden_dim: int = 128
    use_prior_head: bool = True # sum only
    vote_sees_q: bool = True # sum only

    # encoder as a plain dict: {'name': 'patchify', 'ch': ..., 'patch_size': ...}
    # keyword-only kwargs for the encoder class, so adding an encoder
    # argument needs no config change here.
    encoder: dict[str, Any] = field(default_factory=lambda: {
        'name': 'patchify', 'ch': 128, 'patch_size': 5,
    })

def _sobel() -> Tensor:
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    return torch.stack([kx, kx.t().contiguous()]).unsqueeze(1)


class SortOfClevrSyncNet(nn.Module):

    has_rotors = False
    is_syncnet = True

    def __init__(
            self, 
            cfg: SortOfClevrSyncNetConfig,
            data_cfg: SortOfClevrDataConfig,
            q_dim: int = C.QUESTION_SIZE,
            answer_dim: int = C.ANSWER_SIZE
            ) -> None:
    
        super().__init__()
        self.cfg = cfg
        enc = dict(cfg.encoder)
        if enc.pop('name', 'patchify') != 'patchify':
            raise ValueError(
                f'sort-of-clevr syncnet supports the patchify encoder '
                f'only, got: {cfg.encoder.get("name")!r}'
            )
        self.q_dim, self.answer_dim = q_dim, answer_dim

        if cfg.partition == 'views' and cfg.n_modules != N_VIEWS:
            raise ValueError(f'views partition requires n_modules={N_VIEWS}')
        if cfg.partition == 'quadrant' and cfg.n_modules != 4:
            raise ValueError('quadrant partition requires n_modules=4')

        M, d = cfg.n_modules, cfg.module_dim
        self.M, self.d = M, d
        self.T, self.dt = cfg.T, cfg.dt

        # ---- input pathway ------------------------------------------
        # channel width entering the token stream
        if cfg.q_conditioning == 'broadcast_cat':
            enc_ch = int(enc['ch'])
            self.ch = enc_ch + cfg.q_emb_dim
            self.q_enc = nn.Linear(q_dim, cfg.q_emb_dim)
        else:
            enc_ch = int(enc['ch'])
            self.ch = enc_ch
            if cfg.q_conditioning == 'film':
                self.film_gamma = nn.Linear(q_dim, enc_ch)
                self.film_beta = nn.Linear(q_dim, enc_ch)
            else:                                   # 'token'
                self.q_enc = nn.Linear(q_dim, enc_ch)

        n_enc = N_VIEWS if cfg.partition == 'views' else 1
        self.encoders = nn.ModuleList([
            PatchifyEncoder(data_cfg.img_size, enc_ch, int(enc['patch_size']))
            for _ in range(n_enc)])
        self.spatial = self.encoders[0].spatial
        self.norms = nn.ModuleList([
            nn.GroupNorm(8, enc_ch, affine=True) for _ in range(n_enc)])
        if cfg.use_pos_emb:
            self.pos_emb = nn.Parameter(
                0.02 * torch.randn(n_enc, enc_ch, self.spatial, self.spatial))

        if cfg.partition == 'views':
            self.register_buffer('sobel', _sobel())
            ys, xs = torch.meshgrid(
                torch.linspace(0, 1, data_cfg.img_size),
                torch.linspace(0, 1, data_cfg.img_size), indexing='ij')
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
            self.embed_to_query = nn.Linear(d, self.ch)
        self.attn_query = nn.Linear(d, self.ch)
        log_beta = torch.tensor(float(np.log(cfg.beta_init)))
        if cfg.learn_beta:
            self.log_beta = nn.Parameter(log_beta)
        else:
            self.register_buffer('log_beta', log_beta)

        # ---- messages + gates ---------------------------------------
        self.msg_proj = nn.Linear(d, cfg.msg_dim)
        if cfg.msg_agg == 'bus':
            self.bus_norm = nn.LayerNorm(cfg.msg_dim)
        self.cell = nn.GRUCell(self.ch + cfg.msg_dim, d)

        omega0 = cfg.omega_init * torch.linspace(-1.0, 1.0, M)
        if cfg.learn_omega:
            self.omega = nn.Parameter(omega0)
        else:
            self.register_buffer('omega', omega0)
        self.K = nn.Parameter(torch.ones(M, M))
        self.k_mlp = nn.Sequential(
            nn.Linear(2 * d, cfg.k_hidden), nn.GELU(),
            nn.Linear(cfg.k_hidden, 1), nn.Tanh())
        if cfg.gate_mode == 'mlp':
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * d, cfg.k_hidden), nn.GELU(),
                nn.Linear(cfg.k_hidden, 1), nn.Sigmoid())
        if cfg.gate_mode == 'attn':
            self.gate_q = nn.Linear(d, cfg.k_hidden)
            self.gate_k = nn.Linear(d, cfg.k_hidden)

        # ---- readout -------------------------------------------------
        self.n_pairs = M * (M - 1) // 2
        iu = torch.triu_indices(M, M, offset=1)
        self.register_buffer('pair_i', iu[0])
        self.register_buffer('pair_j', iu[1])
        if cfg.readout_mode in ('sync', 'both'):
            self.sync_decay_theta = nn.Parameter(-2.0 * torch.ones(self.n_pairs))
            self.sync_decay_h = nn.Parameter(-2.0 * torch.ones(self.n_pairs))

        self.content_head = nn.Linear(self.ch, cfg.content_dim)
        if cfg.readout_mode == 'sum':
            vote_in = cfg.content_dim + d + (q_dim if cfg.vote_sees_q else 0)
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
            head_in = {'concat': M * cfg.content_dim, 'sync': sync_dim,
                       'both': M * cfg.content_dim + sync_dim}[cfg.readout_mode] + q_dim
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

    # gates

    @staticmethod
    def _gates_from_theta(theta: Tensor) -> Tensor:
        return 0.5 * (1.0 + torch.cos(theta.unsqueeze(-1) - theta.unsqueeze(-2)))

    def _gates(self, theta: Tensor, h: Tensor) -> Tensor:
        mode = self.cfg.gate_mode
        B, M, d = h.shape
        if mode in ('phase', 'frozen'):
            return self._gates_from_theta(theta)
        if mode == 'open':
            return torch.ones(B, M, M, device=h.device)
        if mode == 'attn':
            gq, gk = self.gate_q(h), self.gate_k(h)
            return F.softmax(torch.einsum('bik,bjk->bij', gq, gk)
                             / (gq.shape[-1] ** 0.5), dim=-1)
        hi = h.unsqueeze(2).expand(B, M, M, d)
        hj = h.unsqueeze(1).expand(B, M, M, d)
        return self.gate_mlp(torch.cat([hi, hj], -1)).squeeze(-1)

    def _phase_step(self, theta: Tensor, h: Tensor) -> Tensor:
        B, M, d = h.shape
        hi = h.unsqueeze(2).expand(B, M, M, d)
        hj = h.unsqueeze(1).expand(B, M, M, d)
        k = self.k_mlp(torch.cat([hi, hj], -1)).squeeze(-1)
        diff = theta.unsqueeze(1) - theta.unsqueeze(2)
        return theta + self.dt * (
            self.omega + (self.K * torch.sin(diff) * k).sum(-1))

    def _sync_features(self, th_hist, h_hist, B) -> Tensor:
        if not th_hist:
            return torch.zeros(B, 2 * self.n_pairs, device=self.pair_i.device)
        theta = torch.stack(th_hist, 1); h = torch.stack(h_hist, 1)
        ages = torch.arange(theta.shape[1] - 1, -1, -1,
                            device=theta.device, dtype=theta.dtype)

        def w(p):
            x = torch.exp(-F.softplus(p).unsqueeze(-1) * ages)
            return x / x.sum(-1, keepdim=True)

        dth = theta[:, :, self.pair_i] - theta[:, :, self.pair_j]
        st = torch.einsum('btp,pt->bp', torch.cos(dth), w(self.sync_decay_theta))
        hn = F.normalize(h, dim=-1)
        ch_ = (hn[:, :, self.pair_i] * hn[:, :, self.pair_j]).sum(-1)
        sh = torch.einsum('btp,pt->bp', ch_, w(self.sync_decay_h))
        return torch.cat([st, sh], -1)

    # ------------------------------------------------------------------

    def forward(self, images: Tensor, questions: Tensor,
                t_override: int | None = None, return_trace: bool = False,
                **batch) -> SortOfClevrOutput:
        cfg = self.cfg
        q = questions.float()
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
        theta = (torch.zeros(B, M, device=dev) if cfg.deterministic_phase
                 else 2 * torch.pi * torch.rand(B, M, device=dev))

        beta = self.log_beta.exp()
        if mask is not None:
            pm = mask.to(dev)
            attn = (pm.float() / pm.float().sum(-1, keepdim=True)
                    ).unsqueeze(0).expand(B, M, P)
        else:
            pm = None
            attn = torch.full((B, M, P), 1.0 / P, device=dev)

        need_sync = cfg.readout_mode in ('sync', 'both')
        th_hist: list[Tensor] = []
        h_hist: list[Tensor] = []
        traces = ({'phase': [], 'gates': [], 'attn': [], 'h': []}
                  if return_trace else None)

        g = self._gates(theta, h)                          # T=0 default
        for _ in range(T):
            g = self._gates(theta, h)
            m_all = self.msg_proj(h)
            if cfg.msg_agg == 'bus':
                msg = self.bus_norm(torch.einsum('bij,bjk->bik', g, m_all))
            else:
                norm = 1.0 if cfg.gate_mode == 'attn' else float(M)
                msg = torch.einsum('bij,bjk->bik', g, m_all) / norm

            qv = self.attn_query(h)
            if cfg.use_module_embed:
                qv = qv + self.embed_to_query(self.module_embed).unsqueeze(0)
            queries = F.normalize(qv, dim=-1)
            if per_module:
                keys = F.normalize(flat, dim=-1)           # (B,M,P,ch)
                logits = beta * torch.einsum('bmc,bmpc->bmp', queries, keys)
            else:
                keys = F.normalize(flat, dim=-1)           # (B,P,ch)
                logits = beta * torch.einsum('bmc,bpc->bmp', queries, keys)
            if pm is not None:
                logits = logits.masked_fill(~pm.unsqueeze(0), float('-inf'))
            attn = F.softmax(logits, dim=-1)
            a = (torch.einsum('bmp,bmpc->bmc', attn, flat) if per_module
                 else torch.einsum('bmp,bpc->bmc', attn, flat))

            h = self.cell(torch.cat([a, msg], -1).reshape(B * M, -1),
                          h.reshape(B * M, d)).reshape(B, M, d)
            if cfg.gate_mode == 'phase':
                theta = self._phase_step(theta, h)

            if need_sync:
                th_hist.append(theta); h_hist.append(h)
            if return_trace:
                traces['phase'].append(theta.detach())     # type: ignore
                traces['gates'].append(g.detach())         # type: ignore
                traces['attn'].append(attn.detach())       # type: ignore
                traces['h'].append(h.detach())             # type: ignore

        m_content = (torch.einsum('bmp,bmpd->bmd', attn, b_content) if per_module
                     else torch.einsum('bmp,bpd->bmd', attn, b_content))

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
                pieces.append(self._sync_features(th_hist, h_hist, B))
            pieces.append(q)
            logits = self.head(torch.cat(pieces, -1))

        with torch.no_grad():
            R = torch.stack([torch.cos(theta), torch.sin(theta)], -1
                            ).mean(1).norm(dim=-1).mean().item()
            off = ~torch.eye(M, dtype=torch.bool, device=dev)
            metrics = {'phase_R': R, 'gate_offdiag': g[:, off].mean().item()}
            if cfg.readout_mode == 'sum':
                v = F.normalize(module_logits, dim=-1)
                metrics['vote_agreement'] = torch.einsum(
                    'bmd,bnd->bmn', v, v)[:, off].mean().item()
                if cfg.use_prior_head:
                    metrics['evidence_ratio'] = (
                        module_logits.sum(1).norm(dim=-1).mean()
                        / (prior.norm(dim=-1).mean() + 1e-8)).item()

        return {'logits': logits, 'traces': traces, 'metrics': metrics}

    @classmethod
    def from_config(
        cls, 
        cfg: SortOfClevrSyncNetConfig,
        data_cfg: SortOfClevrDataConfig
        ) -> SortOfClevrSyncNet:

        return cls(cfg, data_cfg)