"""B -- decide which cells belong together and hand each module its
content. L1 lives here: binding without competition collapses to a single
attractor, so every binder either implements competition or is marked as
the control that demonstrates its absence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CompetitiveClaim(nn.Module):
    """The canonical binder. Anchors sampled from a learned prior (shared,
    or per-module when the identity component says so), then mean-shift with
    softmax OVER SLOTS: cells choose exactly one owner. Exclusivity is
    load-bearing (L1); content never enters the competition -- a slot has
    something to say only if the field bound an object."""

    def __init__(self, feat_dim: int, slot_dim: int, n_slots: int,
                 n_groups: int, osc_dim: int, iters: int = 3,
                 beta: float = 8.0, per_module_anchors: bool = False,
                 claim_prior: str | None = None, n_cells: int | None = None,
                 claim_prior_init: str = 'zero', claim_prior_scale: float = 4.0):
        super().__init__()
        self.n_slots, self.iters = n_slots, iters
        # claim_prior='spatial': a learned per-slot bias over cells, added to
        # the competition logits. Slots start hunting in their own region
        # instead of racing from symmetric noise -- partition binding as a
        # PRIOR while competition remains the mechanism. Zero-initialised, so
        # step 0 is the unmodified claim; anneal to zero to end architecturally
        # identical to the canonical binder.
        self.claim_prior = claim_prior
        if claim_prior == 'spatial':
            if n_cells is None:
                raise ValueError("claim_prior='spatial' needs n_cells")
            bias = torch.zeros(n_slots, n_cells)
            if claim_prior_init == 'partition':
                # slot k is predisposed to the k-th cell of a rows x cols grid
                # over the S x S field (K=6 -> 2x3). Extra slots stay unbiased.
                S = int(round(n_cells ** 0.5))
                rows = max(1, int(n_slots ** 0.5)); cols = -(-n_slots // rows)
                ys = torch.arange(S).repeat_interleave(S); xs = torch.arange(S).repeat(S)
                region = (ys * rows // S) * cols + (xs * cols // S)
                for k in range(min(n_slots, rows * cols)):
                    bias[k, region == k] = claim_prior_scale
            elif claim_prior_init == 'random':
                # Control for the partition seed: every cell assigned to a random
                # slot (fixed seed), same bias scale. If THIS trains as well as
                # the grid, any symmetry-breaking seed suffices.
                g = torch.Generator().manual_seed(0)
                region = torch.randint(0, n_slots, (n_cells,), generator=g)
                for k in range(n_slots):
                    bias[k, region == k] = claim_prior_scale
            elif claim_prior_init != 'zero':
                raise ValueError(f"unknown claim_prior_init {claim_prior_init!r}")
            self.cell_bias = nn.Parameter(bias)
            self.register_buffer('prior_scale', torch.ones(()))
        a_m = n_slots if per_module_anchors else 1
        self.anchor_mu = nn.Parameter(torch.randn(1, a_m, n_groups, osc_dim))
        self.anchor_log_sigma = nn.Parameter(torch.zeros(1, a_m, n_groups, osc_dim))
        self.log_beta = nn.Parameter(torch.log(torch.tensor(beta)))
        self.to_slot = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, slot_dim))

    def sample_anchors(self, B: int, device, dtype,
                       prior_perm: Tensor | None = None) -> Tensor:
        mu = self.anchor_mu.expand(1, self.n_slots, -1, -1)
        ls = self.anchor_log_sigma.expand(1, self.n_slots, -1, -1)
        if prior_perm is not None:                                           # anchor_shuffle intervention
            mu, ls = mu[:, prior_perm], ls[:, prior_perm]
        phi = mu + ls.exp() * torch.randn(B, self.n_slots, *self.anchor_mu.shape[2:],
                                          device=device, dtype=dtype)
        return F.normalize(phi, dim=-1)

    def forward(self, feats: Tensor, Zt: Tensor,
                prior_perm: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """feats (B, P, F); Zt (B, P, K, d) -> slots (B, M, slot_dim),
        anchors (B, M, K, d), reads (B, M, P)."""
        K_f = Zt.shape[2]
        B = feats.shape[0]
        phi = self.sample_anchors(B, feats.device, feats.dtype, prior_perm)
        reads = slots = None
        for _ in range(self.iters):
            logits = self.log_beta.exp() * torch.einsum('bkgd,bngd->bkn', phi, Zt) / K_f
            if self.claim_prior == 'spatial':
                logits = logits + self.prior_scale * self.cell_bias.unsqueeze(0)
            attn = F.softmax(logits, dim=1)                                  # cells choose slots
            reads = attn / (attn.sum(-1, keepdim=True) + 1e-8)
            phi = F.normalize(torch.einsum('bkn,bngd->bkgd', reads, Zt), dim=-1)
            slots = self.to_slot(torch.einsum('bkn,bnf->bkf', reads, feats))
        return slots, phi, reads


class QueryRead(nn.Module):
    """The falsified control, kept executable (L1 exhibit). Per-module
    phase-space queries, softmax OVER CELLS: no exclusivity force, so on a
    weakly structured field every reader converges to the same attractor
    (read overlap -> 1) and the image pathway dies at the question floor.
    exclusive=True adds a cells-choose-modules renormalisation -- the
    untested middle, documented as such."""

    def __init__(self, feat_dim: int, slot_dim: int, n_slots: int,
                 n_groups: int, osc_dim: int, query_dim: int,
                 beta: float = 8.0, exclusive: bool = False):
        super().__init__()
        self.n_slots, self.exclusive = n_slots, exclusive
        self.n_groups, self.osc_dim = n_groups, osc_dim
        self.q_phi = nn.Linear(query_dim, n_groups * osc_dim)
        self.log_beta = nn.Parameter(torch.log(torch.tensor(beta)))
        self.to_slot = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, slot_dim))

    def forward(self, feats: Tensor, Zt: Tensor, h_query: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B = feats.shape[0]
        K_f = Zt.shape[2]
        phi = F.normalize(self.q_phi(h_query).view(B, self.n_slots, self.n_groups, self.osc_dim), dim=-1)
        logits = self.log_beta.exp() * torch.einsum('bkgd,bngd->bkn', phi, Zt) / K_f
        attn = F.softmax(logits, dim=-1)                                     # each module reads freely
        if self.exclusive:
            attn = attn / (attn.sum(1, keepdim=True) + 1e-8)
        slots = self.to_slot(torch.einsum('bkn,bnf->bkf', attn, feats))
        return slots, phi, attn


class PartitionRead(nn.Module):
    """Binding by fiat: a fixed spatial partition (a g x g grid of regions,
    quadrants at g = 2), one module per region, content = the region's
    feature mean. The gated model's binder; the baseline the competitive
    claim replaced."""

    def __init__(self, feat_dim: int, slot_dim: int, grid: int = 2):
        super().__init__()
        self.grid = grid
        self.to_slot = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, slot_dim))

    def forward(self, f_map: Tensor) -> Tensor:
        """f_map (B, F, S, S) -> slots (B, grid^2, slot_dim)."""
        B, Fc, S, _ = f_map.shape
        g = self.grid
        pooled = F.adaptive_avg_pool2d(f_map, g)                             # (B, F, g, g)
        return self.to_slot(pooled.flatten(2).transpose(1, 2))


class GivenTokens(nn.Module):
    """Identity binding over ground-truth scene descriptors: the mechanism
    isolator. Same medium downstream, perception assumed away -- this is the
    control that localises pixel failures to perception. SEGREGATION: this
    binder consumes `scenes`, never images; its sole importer is
    token_busnet.py."""

    def __init__(self, token_in: int, slot_dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(token_in), nn.Linear(token_in, slot_dim))

    def forward(self, tokens: Tensor) -> Tensor:
        return self.proj(tokens)
