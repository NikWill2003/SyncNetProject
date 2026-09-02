"""THE SYNCNET, PURE FORM
=========================

The canonical model of the thesis with every ablation switch, alternative
front end, alternative medium, and intervention hook removed: what remains is
exactly the computation of the best tested cell (`sync_d/field_stim6`,
Sort-of-CLEVR from raw 75x75 pixels: .923 / .784 ternary and .914 / .761 on
its two healthy seeds at 100k steps). One file, one path, dependencies only
on torch. Parameter names mirror the full implementation (src/models/
busnet.py with encoder={name: field}, per_module_gru=false, phase_repr=
vector, osc_dim=6, drive=stimulus), so a trained checkpoint of that cell
loads here via load_state_dict(..., strict=False) and produces bit-identical
logits -- verified in tools/check_pure_equivalence.py.

The idea in one paragraph
-------------------------
The network is built from the thesis's two uses of synchrony and nothing
else. Perception binds by synchrony: every cell of a conv feature map
carries oscillators, phase spreads through a learned local coupling until
cells of one object rotate together, and six exchangeable slots win their
objects by phase alignment alone -- no colour keys, no positions, no content
in the competition. Communication is coherence: the six slots and one
question-holding head are rows on a single shared bus; each writes its
message multiplied by its own unit phase vector, the wire carries the sum,
and each reads that one sum through its own orthonormal frame, so an
in-phase sender arrives intact, an orthogonal one cancels. Phases evolve as
driven Kuramoto oscillators whose stimulus each module computes from its own
state, so *what a module represents places it on the medium*. The answer is
read from the head alone, which never sees the image: nothing reaches the
output except what crossed the bus.

The pipeline, stage by stage
----------------------------
1. ENCODER  images (B, 3, 75, 75) -> f (B, 64, 19, 19). Two stride-2 conv
   stages and a 3x3 head; FiLM from the question, GroupNorm, a learned
   positional embedding. The question is allowed into perception exactly
   once, here, as feature modulation.
2. OSCILLATOR FIELD  every position p carries 16 unit 4-vectors u_{p,k} on
   S^3. Eight Euler steps of
       u <- Pi( u + dt_f ( Omega u + P_u( J*u + W_c f ) ) ),
   with Omega a learned per-group antisymmetric generator (natural
   rotation), J a learned 5x5 convolution through which phase propagates
   spatially, W_c f the feature stimulus, P_u the tangent projection and Pi
   renormalisation. After settling, positions of one object share a phase
   that other objects do not: binding by synchrony.
3. PHASE SLOTS  six slots hold anchors phi on the concatenated oscillators.
   Three mean-shift iterations: every cell softmaxes OVER SLOTS on phase
   alignment beta<phi, u_p>, each anchor moves to the renormalised weighted
   mean of the cells it won, and the slot's content X_s is the feature
   average of its cells (LayerNorm + linear to 64-d, FiLM-ed by the
   question). A slot carries information only if the field bound an object.
4. THE BUS  rows = 6 slots + 1 head (initialised from the question, no
   image access). Each row i holds a state h_i (GRU, 96-d, shared cell --
   slots are exchangeable) and a phase z_i on S^5 (d = 6). Per step:
     write   m_i = W_m h_i               (4-d message)
     wire    Bus = sum_j m_j z_j^T       (one 4x6 matrix, shared by all)
     read    r_i = [Bus f_i^(1); ...; Bus f_i^(6)] / N, the bus through row
             i's orthonormal frame (first vector z_i, rest Gram-Schmidt of
             fixed references), own contribution subtracted (echo cancel)
     update  h_i <- GRU([X_i; r_i], h_i)   (head rows: X = 0)
     phase   z_i <- Pi( z_i + dt ( omega_i A z_i
                        + P_z( sum_j K_ij kappa_ij(h) z_j + W_s h_i ) ) )
   kappa_ij(h) is a learned pairwise MLP on the two states (signed content
   coupling); W_s h_i is the STIMULUS: the term that lets a module set its
   own orientation from what it represents, which is the difference between
   addresses that must be discovered by the dynamics and addresses the state
   places directly -- the single most important design choice in the model.
   Slot phases are initialised from their field anchors (the grouping found
   in perception is what the medium starts from); the head's phase starts
   at random and is steered entirely by its own stimulus.
5. READOUT  logits = MLP([h_head^(T); question]) + MLP_prior(question). The
   prior head absorbs what the question alone predicts; everything above the
   question-only floor arrived over the bus.

Why these choices (each is a tested claim, not a preference)
------------------------------------------------------------
- Shared bus, no gate: on private lines a content gate can make every
  routing decision directly, and in every trained-from-scratch comparison
  the optimiser then abandons the phase (the gated SyncNet's collapse, and
  PhaseBind: lambda -> 0). On a bus there is nothing else to use.
- Stimulus drive: with coupling alone the pixel model lands below even
  static addresses (.828 vs .851); with it, computed addresses beat stored
  ones and show the signature that proves it (freezing the dynamics costs
  .77-.82, shuffling the initial phases costs ~0; the static control shows
  the mirror image).
- d = 6: one orthogonal direction available per object; the capacity of the
  medium is the number of senders a receiver can separate at once, and the
  circle (d = 2) provably cannot serve these questions.
- Head-only readout: makes the silent bus the question-only floor by
  construction and every accuracy point attributable to communication.
- Exchangeable slots, shared GRU: per-module parameters or identity
  embeddings would smuggle identity-based routing back in.

Known limits, stated plainly: the slot binding is a lottery (roughly one
seed in three fails to assemble within a 100k cosine budget; coverage < .35
early is the signature), and longer schedules can reorganise the solution
(a 200k cosine found a count-optimised layout trading comparisons for
ternary). Both are properties of training, not of the architecture, and both
are reported, not hidden.

Usage
-----
    model = PureSyncNet(img_size=75, answer_dim=10)
    logits = model(images, questions)['logits']     # questions: (B, 18)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# canonical hyperparameters of the tested cell
M_SLOTS = 6          # modules (slots); one head is added on the bus
D_PHASE = 6          # bus phase dimension: S^{d-1}, one axis per object
D_MODULE = 96        # module state width
D_MSG = 4            # message width on the bus
T_BUS = 8            # communication steps
DT_BUS = 0.5
FIELD_CH = 64        # feature channels entering the field
FIELD_HIDDEN = 48
FIELD_GROUPS = 16    # oscillator groups per position
FIELD_OSC_D = 4      # each group lives on S^3
T_FIELD = 8
DT_FIELD = 1.0
SLOT_ITERS = 3
TOK_DIM = 64
Q_SIZE = 18
HIDDEN = 128
K_HIDDEN = 64


def tangent(z: Tensor, v: Tensor) -> Tensor:
    """Project v onto the tangent space of the unit sphere at z."""
    return v - (v * z).sum(-1, keepdim=True) * z


class FieldEncoder(nn.Module):
    """Two stride-2 stages then a 3x3 head: 75x75 -> 19x19, FIELD_CH channels."""

    def __init__(self, img_size: int, hidden: int = FIELD_HIDDEN, n_down: int = 2, out_ch: int = FIELD_CH):
        super().__init__()
        layers: list[nn.Module] = []
        cin, s = 3, img_size
        for _ in range(n_down):
            layers += [nn.Conv2d(cin, hidden, 3, stride=2, padding=1), nn.GroupNorm(8, hidden), nn.SiLU()]
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

    def __init__(self, fch: int = FIELD_CH, osc_dim: int = FIELD_OSC_D, n_groups: int = FIELD_GROUPS,
                 T: int = T_FIELD, dt: float = DT_FIELD, ksize: int = 5):
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


class PhaseSlotAttention(nn.Module):
    """Slots compete for positions by phase alignment alone. Anchors are
    sampled from a learned distribution, then three mean-shift iterations:
    softmax over slots on beta * mean-group alignment, anchors move to the
    renormalised weighted mean, content is the weighted feature average."""

    def __init__(self, feat_dim: int = FIELD_CH, slot_dim: int = TOK_DIM, n_slots: int = M_SLOTS,
                 n_groups: int = FIELD_GROUPS, osc_dim: int = FIELD_OSC_D, iters: int = SLOT_ITERS,
                 beta: float = 8.0):
        super().__init__()
        self.n_slots, self.iters = n_slots, iters
        self.anchor_mu = nn.Parameter(torch.randn(1, 1, n_groups, osc_dim))
        self.anchor_log_sigma = nn.Parameter(torch.zeros(1, 1, n_groups, osc_dim))
        self.log_beta = nn.Parameter(torch.log(torch.tensor(beta)))
        self.to_slot = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, slot_dim))

    def forward(self, feats: Tensor, Zt: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """feats (B, P, F); Zt (B, P, K, d) unit per group -> slots, anchors, reads."""
        K_f = Zt.shape[2]
        B = feats.shape[0]
        phi = self.anchor_mu + self.anchor_log_sigma.exp() * torch.randn(
            B, self.n_slots, *self.anchor_mu.shape[2:], device=feats.device, dtype=feats.dtype)
        phi = F.normalize(phi, dim=-1)
        reads = None
        for _ in range(self.iters):
            logits = self.log_beta.exp() * torch.einsum('bkgd,bngd->bkn', phi, Zt) / K_f
            attn = F.softmax(logits, dim=1)                                  # cells choose slots
            reads = attn / (attn.sum(-1, keepdim=True) + 1e-8)
            phi = F.normalize(torch.einsum('bkn,bngd->bkgd', reads, Zt), dim=-1)   # mean-shift on the sphere
            slots = self.to_slot(torch.einsum('bkn,bnf->bkf', reads, feats))
        return slots, phi, reads


class SkewGenerator(nn.Module):
    """A learned antisymmetric generator: omega * A z rotates z at rate
    omega. A is normalised so its largest rotation rate is 1, making omega
    an angular frequency."""

    def __init__(self, d: int = D_PHASE, init_scale: float = 0.1):
        super().__init__()
        self.raw = nn.Parameter(init_scale * torch.randn(d, d))

    def forward(self, z: Tensor, omega: Tensor) -> Tensor:
        A = self.raw - self.raw.t()
        A = A / (A.norm() / math.sqrt(2) + 1e-6)
        return omega.unsqueeze(-1) * torch.einsum('de,bne->bnd', A, z)


class PureSyncNet(nn.Module):
    """The canonical SyncNet: field perception, phase slots, stimulus-driven
    vector-phase bus, head readout. See the module docstring."""

    def __init__(self, img_size: int = 75, answer_dim: int = 10, q_size: int = Q_SIZE):
        super().__init__()
        M, dm, d = M_SLOTS, D_MODULE, D_PHASE
        self.M, self.dm, self.d, self.T = M, dm, d, T_BUS
        self.N = M + 1                                                       # slots + the head
        self.n_rx = d

        # perception -----------------------------------------------------
        self.field_enc = FieldEncoder(img_size)
        S = self.field_enc.spatial
        self.grid_film_gamma = nn.Linear(q_size, FIELD_CH)
        self.grid_film_beta = nn.Linear(q_size, FIELD_CH)
        self.grid_norm = nn.GroupNorm(8, FIELD_CH, affine=True)
        self.pos_emb = nn.Parameter(0.02 * torch.randn(1, FIELD_CH, S, S))
        self.field = OscillatorField()
        self.phase_slots = PhaseSlotAttention()
        self.anchor_to_phase = nn.Linear(FIELD_GROUPS * FIELD_OSC_D, d)

        # slot content conditioning --------------------------------------
        self.film_gamma = nn.Linear(q_size, TOK_DIM)
        self.film_beta = nn.Linear(q_size, TOK_DIM)
        self.norm = nn.LayerNorm(TOK_DIM)

        # modules and the head -------------------------------------------
        self.h_init = nn.Sequential(nn.Linear(q_size, 64), nn.GELU(), nn.Linear(64, M * dm))
        self.module_embed = nn.Parameter(torch.randn(M, dm) / dm ** 0.5)     # unused by the field path; kept for checkpoint compatibility
        self.head_init = nn.Sequential(nn.Linear(q_size, 64), nn.GELU(), nn.Linear(64, dm))
        self.head_embed = nn.Parameter(torch.randn(1, dm) / dm ** 0.5)
        self.cell = nn.GRUCell(TOK_DIM + D_MSG * self.n_rx, dm)

        # the medium ------------------------------------------------------
        self.msg_proj = nn.Linear(dm, D_MSG)
        g = torch.Generator().manual_seed(1234)
        self.register_buffer('frame_ref', F.normalize(torch.randn(d - 1, d, generator=g), dim=-1))

        # phase dynamics --------------------------------------------------
        self.omega = nn.Parameter(torch.zeros(self.N))
        self.K = nn.Parameter(torch.ones(self.N, self.N))
        self.k_mlp = nn.Sequential(nn.Linear(2 * dm, K_HIDDEN), nn.GELU(), nn.Linear(K_HIDDEN, 1), nn.Tanh())
        self.stim = nn.Linear(dm, d)
        self.gen = SkewGenerator(d)

        # readout ---------------------------------------------------------
        self.head_out = nn.Sequential(nn.Linear(dm + q_size, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, answer_dim))
        self.prior_head = nn.Sequential(nn.Linear(q_size, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, answer_dim))

    # -- the medium ------------------------------------------------------
    def _frame(self, z: Tensor) -> Tensor:
        """(B, N, d) -> (B, N, d, d): row i's frame, first vector z_i, the
        rest Gram-Schmidt of fixed reference directions."""
        vecs = [z]
        for k in range(self.n_rx - 1):
            v = self.frame_ref[k].to(z.dtype).expand_as(z)
            for u in vecs:
                v = v - (v * u).sum(-1, keepdim=True) * u
            vecs.append(F.normalize(v, dim=-1))
        return torch.stack(vecs, 2)

    def _receive(self, h: Tensor, z: Tensor) -> Tensor:
        """One shared wire: write m_j z_j^T, sum, read through own frame,
        subtract own echo."""
        B, N, _ = h.shape
        m = self.msg_proj(h)                                                 # (B, N, D)
        bus = torch.einsum('bnD,bnd->bDd', m, z)                             # shared by every receiver
        Fr = self._frame(z)                                                  # (B, N, d, d)
        r = torch.einsum('bDd,bnad->bnDa', bus, Fr)
        r = r - torch.einsum('bnD,bnd,bnad->bnDa', m, z, Fr)                 # echo cancellation
        return r.flatten(2) / float(N)

    # -- phase dynamics ---------------------------------------------------
    def _z_step(self, z: Tensor, h: Tensor) -> Tensor:
        B, N, dm = h.shape
        vel = self.gen(z, self.omega.to(z.dtype).unsqueeze(0).expand(B, N))
        hi = h.unsqueeze(2).expand(B, N, N, dm)
        hj = h.unsqueeze(1).expand(B, N, N, dm)
        kap = self.k_mlp(torch.cat([hi, hj], -1)).squeeze(-1)                # signed content coupling in [-1, 1]
        pull = torch.einsum('bij,bjd->bid', self.K.to(z.dtype).unsqueeze(0) * kap, z)
        vel = vel + tangent(z, pull)
        vel = vel + tangent(z, self.stim(h))                                 # the stimulus drive
        return F.normalize(z + DT_BUS * vel, dim=-1)

    # -- forward ----------------------------------------------------------
    def forward(self, images: Tensor, questions: Tensor) -> dict:
        B, M = images.shape[0], self.M
        questions = questions.float()
        squeeze = questions.dim() == 2
        if squeeze:
            questions = questions.unsqueeze(1)                               # (B, 1, 18)
        q_all = questions.reshape(B, -1)

        # perception: field -> phase slots
        f = self.field_enc(images)
        f = f * (1 + self.grid_film_gamma(q_all))[..., None, None] + self.grid_film_beta(q_all)[..., None, None]
        f = self.grid_norm(f) + self.pos_emb
        zf = self.field(f)
        Zt = self.field.to_tokens(zf)                                        # (B, P, K, d_f)
        feats = f.flatten(2).transpose(1, 2)                                 # (B, P, fch)
        X, anchors, reads = self.phase_slots(feats, Zt)                      # (B, M, tok)
        X = self.norm(X * (1 + self.film_gamma(q_all)).unsqueeze(1) + self.film_beta(q_all).unsqueeze(1))

        # rows on the bus: slots + the head (no image access, X = 0)
        h = self.h_init(q_all).reshape(B, M, self.dm)
        h_head = self.head_init(questions) + self.head_embed.unsqueeze(0)
        h = torch.cat([h, h_head], 1)
        X = torch.cat([X, torch.zeros(B, 1, X.shape[-1], device=X.device, dtype=X.dtype)], 1)

        # phases: slots start from their field anchors, the head at random
        z = F.normalize(torch.randn(B, self.N, self.d, device=X.device, dtype=X.dtype), dim=-1)
        z = torch.cat([F.normalize(self.anchor_to_phase(anchors.flatten(2)), dim=-1), z[:, M:]], 1)

        # communicate
        for _ in range(self.T):
            r = self._receive(h, z)
            inp = torch.cat([X, r], -1)
            h = self.cell(inp.reshape(B * self.N, -1), h.reshape(B * self.N, self.dm)).reshape(B, self.N, self.dm)
            z = self._z_step(z, h)

        # readout: the head, plus the question prior
        logits = self.head_out(torch.cat([h[:, M:], questions], -1))
        logits = logits + self.prior_head(questions)
        if squeeze:
            logits = logits.squeeze(1)
        return {'logits': logits, 'phases': z, 'slot_reads': reads}
