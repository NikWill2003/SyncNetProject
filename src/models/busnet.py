"""BusNet: phase-coded multiplexing on a shared medium.

Six object modules (one per object, read off the pixels), which have to
answer several pairwise questions at once. Messages do not travel on
per-sender channels; they are written on ONE bus tagged by the sender's
phase, and each receiver demodulates the bus with its own phase:

    bus        B      = sum_j m_j e^{i theta_j}          (two real channels)
    receive    r_i    = Re( e^{-i theta_i} B )  =  sum_j m_j cos(theta_j - theta_i)

so an in-phase sender arrives intact, an antiphase sender arrives
NEGATED (not removed), and an orthogonal sender (90 degrees) cancels.
"Closed" on this bus is orthogonality, so the circle offers exactly two
mutually silent channels (in-phase and quadrature): two conversations
is the capacity of S^1, and a third necessarily leaks by cos 60 = 0.5. This is the thesis
gate with the (1+cos)/2 offset removed, and it is what
communication-through-coherence claims: a shared medium, selective
reception by phase locking. Nothing else can select a sender here --
there is no per-sender access, so an attention gate cannot exist on the
bus. The comparators are the open bus (theta = 0 for everyone: a plain
sum, i.e. interference), the silent bus (zero), and, as an upper bound
with per-sender access restored, the ordinary attention gate.

With two disjoint conversations on the bus, the phase solution is two
groups 90 degrees apart, the one multi-group structure S^1 separates
without leakage.

Two details keep the alternative routes closed. The HEAD readout: each
question gets a head module with no object, only the question, its own
phase, and the same demodulated bus signal every other module gets; the
answer is read from its final state. Nothing reaches the answer except
through the medium, so the answering coalition is {head, A, B} and both
objects must broadcast at the head's phase. (The 'asker' readout reads
module A's state directly, a private line for the last hop; a sum-over-
modules readout answers left_of with no communication at all: A votes
+x_A, B votes -x_B, and the sum is the comparison.) And a NARROW bus (msg_dim=1 by default): each sender
needs one scalar per question, so with two questions a 2-dim bus can be
split by dimension per role, and only a 1-dim bus forces phase-division.
Sweep msg_dim in {1, 2, 4}: the phase advantage should appear at 1 and
vanish by 4.

Phase formation: Kuramoto with a signed content coupling kappa(h)
(`coupling='mlp'`), and/or a stimulus that lets a module set its own
phase from its state (`drive='stimulus'`, it knows which question names
it). theta_init random | learned | zero.

forward(images, questions (B, n_q, 18)) -> logits (B, n_q, answer_dim);
a 2-d (B, 18) question (the Sort-of-CLEVR contract) is treated as n_q = 1
and returns (B, answer_dim), so the model runs on the standard task.

On the standard task the demand is different from the pairwise one: a
binary question needs the head to hear five objects over one wire, which
a single phase cannot separate at once. The model has to multiplex in
TIME -- senders at different natural frequencies drift through alignment
with the head on different steps (the omega term), or the head sweeps --
or the senders must take turns being loud. That is what to look for in
head_tvar / the phase-trajectory figure. On this task msg_dim is not
forced to 1 (a sender may need position and shape at once); sweep it.
Metrics on the standard task: head_own_align (head vs the queried
object(s)), head_other_align (head vs the rest), head_tvar (does the
head's alignment pattern change over steps). test_binding/coalition_score
is not meaningful here (the head row has no object).
readout: head | asker | sum.  Overrides: phase_override {zero (open bus), shuffle, freeze}.

PIXELS, all-synchrony (encoder: {name: field}): an oscillator field over
the image (OscField's dynamics) and PHASE slot attention: K exchangeable
slots compete for cells by phase alignment with their anchors and update
the anchors by mean-shift on the sphere; a slot's content is the feature
average of the cells it won, and its bus phase is initialised from its
anchor. Binding by synchrony in perception, communication by synchrony on
the bus; nothing but phase groups or routes anything. Control:
slot_read=both (a content term in the competition).

PIXELS, no identities (encoder: {name: slots, ...}): a CNN grid and slot
attention (Locatello et al. 2020) with K = n_modules exchangeable slots
that compete for the cells; the slots are the modules. No colour keys, no
partition, no per-module identity, so use per_module_gru=false. The
slot masks are exported as the read trace, so obj_coverage / module_purity
measure whether the slots found the objects.

PIXELS, colour-keyed (encoder: {name: cnn|patchify, ...}): instead of the exact object
descriptors, each module keeps its colour identity and LEARNS to read its
object from a conv grid, attending over the tokens with a query built from
its identity embedding and its state; the attended token (content plus
positional embedding) is its input. Perception is learned end-to-end, the
medium is unchanged. The head reads nothing from the image. The attention
maps are exported so test_binding/obj_coverage and module_purity report
whether module k actually reads object k.

PHASE DIMENSION (phase_repr='vector', osc_dim=d): each row carries a unit
vector z in S^{d-1} instead of an angle. A sender writes m_j (x) z_j onto
the bus, and a receiver reads it through a RECEIVER-CENTRED orthonormal
frame -- its own z_i completed to a basis by Gram-Schmidt against fixed
generic reference vectors -- giving d channels,

    r_i^{(a)} = sum_j m_j <z_j, f_i^{(a)}>,   f_i^{(1)} = z_i,

the generalisation of in-phase / quadrature reading (which is the d=2
case). So d is the medium's simultaneous capacity: on the circle two
senders can be separated at once, on S^{d-1} d of them, and at d >= the
number of senders the bus degenerates into private lines (a fixed axis
per sender). Coalitions showed the circle cannot realise frustrated
graphs and d=3 can; on a bus the same knob sets how many senders a head
can hear at once, which is what the relational questions need (5-6).
phase_repr='angle' (default) is the original scalar implementation and
is left untouched so earlier runs stay comparable; use 'vector' for the
whole d-sweep, including d=2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.config import ModelConfig
from .object_tokens import ObjectTokenizer
from .encoders import build_encoder
from .osc_core import FieldEncoder, OscillatorField
from .oscillators import SkewGenerator, random_unit, sphere_step, tangent


@dataclass
class BusNetConfig(ModelConfig):
    n_questions: int = 1                # 1 = the task's own questions; 2+ = pairwise multi-question
    q_size: int = 18
    obj_size: int = 5
    tok_dim: int = 64
    # front-end: exact object descriptors, or colour-keyed slots that learn to
    # read their object from a conv grid (module k = colour k either way)
    encoder: dict[str, Any] = field(default_factory=lambda: {'name': 'objects', 'ch': 128, 'hidden': 64, 'patch_size': 5})
    read_beta_init: float = 5.0
    # colour-keyed grid read upgrades (defaults reproduce the one-shot single-query read)
    read_heads: int = 1                 # queries per module, concatenated then projected
    read_norm: str = 'tokens'           # tokens | modules (competition: cells choose a module)
    read_every_step: bool = False       # re-read the grid from the evolving state at every step
    use_pos_emb: bool = True
    slot_iters: int = 3                 # slots / field front-ends
    # field front-end (encoder: {name: field}): OscField dynamics, then PHASE
    # slot attention -- slots compete for cells by phase alignment and update
    # their anchors by mean-shift on the sphere; content is the feature
    # average of the cells a slot won. slot_read=phase uses phase only (the
    # field must segregate objects for the slots to carry anything);
    # slot_read=both adds a content term (the control).
    field_hidden: int = 48
    field_n_down: int = 2
    field_ch: int = 64
    field_osc_dim: int = 4
    field_groups: int = 16
    field_T: int = 8
    field_dt: float = 1.0
    field_ksize: int = 5
    field_coupling: str = 'conv'
    field_stimulus: bool = True
    slot_read: str = 'phase'            # phase | both
    slot_beta: float = 8.0
    slot_phase_from_anchor: bool = True # a module's bus phase starts from its object's field phase
    n_modules: int = 6                  # = number of objects
    module_dim: int = 96
    per_module_gru: bool = True
    T: int = 8
    dt: float = 0.5
    # medium
    medium: str = 'bus'                 # bus | channels
    bus_phase: str = 'phase'            # bus: phase | open | zero
    quadrature: bool = True             # also read the sin channel (I/Q): two clean channels per step
    echo_cancel: bool = True            # a receiver subtracts its own message
    gate_mode: str = 'attn'             # channels: attn | open | zero | full (all senders concatenated)
    msg_dim: int = 4
    # phase dynamics
    coupling: str = 'mlp'               # mlp | none
    drive: str = 'none'                 # none | stimulus
    theta_init: str = 'random'          # random | learned | zero
    phase_repr: str = 'angle'           # angle (scalar theta, original) | vector (z on S^{d-1})
    # exploratory levers (defaults = exactly the current model)
    phase_step_max_deg: float = 0.0     # cap the per-step rotation of a bus phase (0 = off)
    stim_ema: float = 0.0               # EMA of the stimulus direction across steps (0 = off)
    head_taps: int = 1                  # head rows per question; >1 = parallel listening phases
    field_step_max_deg: float = 0.0     # cap the per-step rotation of a field oscillator (0 = off)
    osc_dim: int = 2                    # vector only: d
    rx_channels: int = 0                # vector only: frame vectors read (0 = all d)
    omega_init: float = 0.5
    learn_omega: bool = True
    k_hidden: int = 64
    # readout
    readout: str = 'head'               # head | asker | sum
    head_transmits: bool = True         # head modules also write to the bus
    hidden_dim: int = 128
    use_prior_head: bool = True


class SlotAttention(nn.Module):
    """Locatello et al. (2020): K exchangeable slots initialised from a learned
    Gaussian, refined by `iters` rounds of competitive attention (softmax over
    slots per input cell, weighted mean over cells) and a shared GRU + MLP.
    Returns slots (B, K, D) and each slot's read distribution over the cells
    (B, K, N), rows summing to one (the weighted-mean weights)."""

    def __init__(self, in_dim: int, slot_dim: int, n_slots: int, iters: int = 3, hidden: int = 128):
        super().__init__()
        self.n_slots, self.iters, self.slot_dim = n_slots, iters, slot_dim
        self.mu = nn.Parameter(torch.randn(1, 1, slot_dim) * 0.1)
        self.log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        self.norm_in = nn.LayerNorm(in_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(nn.Linear(slot_dim, hidden), nn.ReLU(), nn.Linear(hidden, slot_dim))

    def forward(self, inputs: Tensor):
        B, N, _ = inputs.shape
        x = self.norm_in(inputs)
        k, v = self.to_k(x), self.to_v(x)                                  # (B, N, D)
        slots = self.mu + self.log_sigma.exp() * torch.randn(B, self.n_slots, self.slot_dim,
                                                             device=inputs.device, dtype=inputs.dtype)
        w = None
        for _ in range(self.iters):
            prev = slots
            q = self.to_q(self.norm_slots(slots))                          # (B, K, D)
            logits = torch.einsum('bkd,bnd->bkn', q, k) / self.slot_dim ** 0.5
            attn = F.softmax(logits, dim=1)                                # cells choose slots
            w = attn / (attn.sum(-1, keepdim=True) + 1e-8)                 # weighted mean per slot
            updates = torch.einsum('bkn,bnd->bkd', w, v)
            slots = self.gru(updates.reshape(-1, self.slot_dim), prev.reshape(-1, self.slot_dim)).view(B, self.n_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, w


class PhaseSlotAttention(nn.Module):
    """Slots compete for cells by PHASE alignment. Anchors phi_k (K_f groups
    of unit d_f-vectors, one per slot) start from a learned distribution;
    each iteration: w = softmax_k(beta * mean_g <phi_k, z_p> [+ content]),
    phi_k <- normalise(sum_p w_kp z_p) per group (mean-shift on the sphere),
    then the slot content is the w-weighted feature average. Returns
    slots (B, K, D), anchors (B, K, K_f, d_f), reads (B, K, N) (rows sum to 1)."""

    def __init__(self, feat_dim: int, slot_dim: int, n_slots: int, n_groups: int, osc_dim: int,
                 iters: int = 3, beta: float = 8.0, content: bool = False):
        super().__init__()
        self.n_slots, self.iters, self.content = n_slots, iters, content
        self.anchor_mu = nn.Parameter(torch.randn(1, 1, n_groups, osc_dim))
        self.anchor_log_sigma = nn.Parameter(torch.zeros(1, 1, n_groups, osc_dim))
        self.log_beta = nn.Parameter(torch.tensor(float(np.log(beta))))
        self.to_slot = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, slot_dim))
        if content:
            self.q_mu = nn.Parameter(torch.randn(1, 1, slot_dim) * 0.1)
            self.to_q = nn.Linear(slot_dim, feat_dim, bias=False)
            self.to_k = nn.Linear(feat_dim, feat_dim, bias=False)
            self.log_beta_c = nn.Parameter(torch.tensor(float(np.log(5.0))))

    def forward(self, feats: Tensor, Zt: Tensor):
        """feats (B, N, F); Zt (B, N, K_f, d_f) unit per group."""
        B, N, _ = feats.shape
        K_f = Zt.shape[2]
        phi = self.anchor_mu + self.anchor_log_sigma.exp() * torch.randn(
            B, self.n_slots, *self.anchor_mu.shape[2:], device=feats.device, dtype=feats.dtype)
        phi = F.normalize(phi, dim=-1)
        slots = torch.zeros(B, self.n_slots, self.to_slot[1].out_features, device=feats.device, dtype=feats.dtype)
        reads = None
        for _ in range(self.iters):
            logits = self.log_beta.exp() * torch.einsum('bkgd,bngd->bkn', phi, Zt) / K_f
            if self.content:
                q = F.normalize(self.to_q(slots + self.q_mu), dim=-1)
                kk = F.normalize(self.to_k(feats), dim=-1)
                logits = logits + self.log_beta_c.exp() * torch.einsum('bkf,bnf->bkn', q, kk)
            attn = F.softmax(logits, dim=1)                                 # cells choose slots
            reads = attn / (attn.sum(-1, keepdim=True) + 1e-8)              # (B, K, N)
            phi = F.normalize(torch.einsum('bkn,bngd->bkgd', reads, Zt), dim=-1)   # mean-shift on the sphere
            slots = self.to_slot(torch.einsum('bkn,bnf->bkf', reads, feats))
        return slots, phi, reads


class BusNet(nn.Module):

    is_syncnet = True
    objects = True
    SUPPORTED_OVERRIDES = frozenset({'t_override', 'phase_override', 'return_trace'})
    GATE_OVERRIDES = frozenset()
    PHASE_OVERRIDES = frozenset({'zero', 'shuffle', 'freeze', 'freeze_field'})

    def __init__(self, cfg: BusNetConfig, img_size: int, answer_dim: int,
                 object_colours: list[tuple[int, int, int]]) -> None:
        super().__init__()
        self.cfg = cfg
        M, dm, Nq = cfg.n_modules, cfg.module_dim, cfg.n_questions
        self.M, self.dm, self.Nq, self.T = M, dm, Nq, cfg.T
        self.taps = max(1, cfg.head_taps) if cfg.readout == 'head' else 1
        self.n_heads = Nq * self.taps if cfg.readout == 'head' else 0
        self.N = M + self.n_heads                       # rows on the bus
        q_all = Nq * cfg.q_size
        self.obj_tok = ObjectTokenizer(object_colours, img_size, cfg.obj_size)
        enc = dict(cfg.encoder)
        name = enc.get('name', 'objects')
        self.objects = name == 'objects'
        if self.objects and self.obj_tok.n_objects != M:
            raise ValueError('n_modules must equal the number of objects on the objects front end')
        self.slots = name == 'slots'
        self.field_fe = name == 'field'
        if self.field_fe:
            self.field_enc = FieldEncoder(img_size, cfg.field_hidden, cfg.field_n_down, cfg.field_ch)
            self.spatial = self.field_enc.spatial
            fch = cfg.field_ch
            self.grid_norm = nn.GroupNorm(8, fch, affine=True)
            if cfg.use_pos_emb:
                self.pos_emb = nn.Parameter(0.02 * torch.randn(1, fch, self.spatial, self.spatial))
            self.grid_film_gamma = nn.Linear(Nq * cfg.q_size, fch)
            self.grid_film_beta = nn.Linear(Nq * cfg.q_size, fch)
            self.field = OscillatorField(fch, cfg.field_osc_dim, cfg.field_groups, cfg.field_T, cfg.field_dt,
                                         cfg.field_ksize, cfg.field_coupling, cfg.field_stimulus, True, 0.1, 'feature',
                                         step_max_deg=cfg.field_step_max_deg)
            self.phase_slots = PhaseSlotAttention(fch, cfg.tok_dim, M, cfg.field_groups, cfg.field_osc_dim,
                                                  cfg.slot_iters, cfg.slot_beta, content=(cfg.slot_read == 'both'))
            if cfg.slot_read not in ('phase', 'both'):
                raise ValueError(f'unknown slot_read {cfg.slot_read!r}')
        elif self.slots:
            enc = dict(enc); enc['name'] = 'cnn'
            enc_ch = int(enc['ch'])
            self.encoder = build_encoder(enc, img_size)
            self.spatial = self.encoder.spatial
            self.grid_norm = nn.GroupNorm(8, enc_ch, affine=True)
            if cfg.use_pos_emb:
                self.pos_emb = nn.Parameter(0.02 * torch.randn(1, enc_ch, self.spatial, self.spatial))
            self.grid_film_gamma = nn.Linear(Nq * cfg.q_size, enc_ch)
            self.grid_film_beta = nn.Linear(Nq * cfg.q_size, enc_ch)
            self.slot_attn = SlotAttention(enc_ch, cfg.tok_dim, M, cfg.slot_iters)
        elif not self.objects:
            enc_ch = int(enc['ch'])
            self.encoder = build_encoder(enc, img_size)
            self.spatial = self.encoder.spatial
            self.grid_norm = nn.GroupNorm(8, enc_ch, affine=True)
            if cfg.use_pos_emb:
                self.pos_emb = nn.Parameter(0.02 * torch.randn(1, enc_ch, self.spatial, self.spatial))
            self.grid_film_gamma = nn.Linear(Nq * cfg.q_size, enc_ch)
            self.grid_film_beta = nn.Linear(Nq * cfg.q_size, enc_ch)
            if cfg.read_norm not in ('tokens', 'modules'):
                raise ValueError(f'unknown read_norm {cfg.read_norm!r}')
            self.read_query = nn.Linear(dm, enc_ch * max(1, cfg.read_heads))
            self.log_read_beta = nn.Parameter(torch.tensor(float(np.log(cfg.read_beta_init))))
            self.grid_to_tok = nn.Linear(enc_ch * max(1, cfg.read_heads), cfg.tok_dim)
        else:
            self.spatial = None
        if self.n_heads:
            self.head_init = nn.Sequential(nn.Linear(cfg.q_size, 64), nn.GELU(), nn.Linear(64, dm))
            self.head_embed = nn.Parameter(torch.randn(Nq * self.taps, dm) / dm ** 0.5)
        if self.objects:
            self.obj_embed = nn.Linear(self.obj_tok.feat_dim, cfg.tok_dim)
        if (self.slots or self.field_fe) and cfg.per_module_gru:
            raise ValueError("slots are exchangeable: use per_module_gru=false with the slots / field front-ends")
        self.film_gamma = nn.Linear(q_all, cfg.tok_dim)
        self.film_beta = nn.Linear(q_all, cfg.tok_dim)
        self.norm = nn.LayerNorm(cfg.tok_dim)

        self.h_init = nn.Sequential(nn.Linear(q_all, 64), nn.GELU(), nn.Linear(64, M * dm))
        self.module_embed = nn.Parameter(torch.randn(M, dm) / dm ** 0.5)
        N = self.N

        # messages
        self.msg_proj = nn.Linear(dm, cfg.msg_dim)
        if cfg.medium == 'channels' and cfg.gate_mode == 'attn':
            self.gate_q = nn.Linear(dm, cfg.k_hidden)
            self.gate_k = nn.Linear(dm, cfg.k_hidden)
        elif cfg.medium == 'channels' and cfg.gate_mode not in ('open', 'zero', 'full'):
            raise ValueError(f'unknown gate_mode {cfg.gate_mode!r}')
        elif cfg.medium == 'bus' and cfg.bus_phase not in ('phase', 'open', 'zero'):
            raise ValueError(f'unknown bus_phase {cfg.bus_phase!r}')
        elif cfg.medium not in ('bus', 'channels'):
            raise ValueError(f'unknown medium {cfg.medium!r}')
        self.vector = cfg.phase_repr == 'vector'
        if cfg.phase_repr not in ('angle', 'vector'):
            raise ValueError(f'unknown phase_repr {cfg.phase_repr!r}')
        if self.vector and cfg.osc_dim < 2:
            raise ValueError('osc_dim must be >= 2')
        self.d = cfg.osc_dim if self.vector else 2
        self.n_rx = (cfg.rx_channels or self.d) if self.vector else (2 if cfg.quadrature else 1)
        if self.vector and not 1 <= self.n_rx <= self.d:
            raise ValueError('rx_channels must be in [1, osc_dim]')
        if cfg.medium == 'bus':
            rx = cfg.msg_dim * self.n_rx
        elif cfg.gate_mode == 'full':
            rx = cfg.msg_dim * self.N                 # every sender, its own slot
        else:
            rx = cfg.msg_dim
        cell_in = cfg.tok_dim + rx
        if self.field_fe and cfg.slot_phase_from_anchor:
            self.anchor_to_phase = nn.Linear(cfg.field_groups * cfg.field_osc_dim, self.d if self.vector else 2)
        if cfg.per_module_gru:
            self.cells = nn.ModuleList([nn.GRUCell(cell_in, dm) for _ in range(N)])
        else:
            self.cell = nn.GRUCell(cell_in, dm)

        # phase (one oscillator per row on the bus, heads included)
        omega0 = cfg.omega_init * torch.linspace(-1.0, 1.0, N)
        if cfg.learn_omega:
            self.omega = nn.Parameter(omega0)
        else:
            self.register_buffer('omega', omega0)
        self.K = nn.Parameter(torch.ones(N, N))
        if cfg.coupling == 'mlp':
            self.k_mlp = nn.Sequential(nn.Linear(2 * dm, cfg.k_hidden), nn.GELU(),
                                       nn.Linear(cfg.k_hidden, 1), nn.Tanh())
        elif cfg.coupling != 'none':
            raise ValueError(f'unknown coupling {cfg.coupling!r}')
        if cfg.drive == 'stimulus':
            self.stim = nn.Linear(dm, self.d if self.vector else 2)
        elif cfg.drive != 'none':
            raise ValueError(f'unknown drive {cfg.drive!r}')
        if cfg.theta_init == 'learned':
            if self.vector:
                self.z0 = nn.Parameter(torch.randn(N, self.d))
            else:
                self.theta0 = nn.Parameter(2 * np.pi * torch.rand(N))
        if self.vector:
            self.gen = SkewGenerator(self.d, learn=True)
            # fixed generic reference directions for the receiver-centred frame
            g = torch.Generator().manual_seed(1234)
            self.register_buffer('frame_ref', F.normalize(torch.randn(self.d - 1, self.d, generator=g), dim=-1))

        # readout: a head module per question listens on the bus; or the asker
        # (first-named module) answers; or every module votes
        if cfg.readout == 'head':
            self.head_out = nn.Sequential(nn.Linear(dm * self.taps + cfg.q_size, cfg.hidden_dim), nn.GELU(),
                                          nn.Linear(cfg.hidden_dim, answer_dim))
        elif cfg.readout == 'asker':
            self.asker_head = nn.Sequential(nn.Linear(dm + cfg.q_size, cfg.hidden_dim), nn.GELU(),
                                            nn.Linear(cfg.hidden_dim, answer_dim))
        elif cfg.readout == 'sum':
            self.vote_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(dm + cfg.q_size, cfg.hidden_dim), nn.GELU(),
                              nn.Linear(cfg.hidden_dim, answer_dim)) for _ in range(M)])
        else:
            raise ValueError(f'unknown readout {cfg.readout!r}')
        if cfg.use_prior_head:
            self.prior_head = nn.Sequential(nn.Linear(cfg.q_size, cfg.hidden_dim), nn.GELU(),
                                            nn.Linear(cfg.hidden_dim, answer_dim))

    # ------------------------------------------------------------------

    # ---------------- vector phases on S^{d-1} ----------------

    def _grid_read(self, h_mod: Tensor, tokens: Tensor, keys: Tensor) -> tuple[Tensor, Tensor]:
        """Colour-keyed read: each module queries the grid from its state
        (+ identity, which lives in the state). Multiple queries per module are
        concatenated; read_norm='modules' makes cells choose a module."""
        cfg = self.cfg
        B, M, _ = h_mod.shape
        H = max(1, cfg.read_heads)
        ch = keys.shape[-1]
        q = F.normalize(self.read_query(h_mod).view(B, M, H, ch), dim=-1)
        logits = self.log_read_beta.exp() * torch.einsum('bmhc,bpc->bmhp', q, keys)
        if cfg.read_norm == 'modules':
            w = F.softmax(logits, dim=1)                                      # cells choose modules
            attn = w / (w.sum(-1, keepdim=True) + 1e-6)
        else:
            attn = F.softmax(logits, dim=-1)
        X = self.grid_to_tok(torch.einsum('bmhp,bpc->bmhc', attn, tokens).reshape(B, M, H * ch))
        return X, attn.mean(2)

    def _init_z(self, B: int, dev, dtype) -> Tensor:
        N, d = self.N, self.d
        if self.cfg.theta_init == 'zero':
            z = torch.zeros(B, N, d, device=dev, dtype=dtype); z[..., 0] = 1.0
            return z
        if self.cfg.theta_init == 'learned':
            return F.normalize(self.z0, dim=-1).unsqueeze(0).expand(B, N, d).to(dtype)
        return random_unit(B, N, d, device=dev, dtype=dtype)

    def _frame(self, z: Tensor) -> Tensor:
        """Receiver-centred orthonormal frame: (B, N, d) -> (B, N, n_rx, d),
        first vector z itself, the rest Gram-Schmidt of fixed references."""
        vecs = [z]
        for k in range(self.n_rx - 1):
            v = self.frame_ref[k].to(z.dtype).expand_as(z)
            for u in vecs:
                v = v - (v * u).sum(-1, keepdim=True) * u
            vecs.append(F.normalize(v, dim=-1))
        return torch.stack(vecs, 2)

    def _z_step(self, z: Tensor, h: Tensor) -> Tensor:
        cfg = self.cfg
        vel = self.gen(z, self.omega.to(z.dtype).unsqueeze(0).expand(z.shape[:2]))
        if cfg.coupling == 'mlp':
            B, N, dm = h.shape
            hi = h.unsqueeze(2).expand(B, N, N, dm); hj = h.unsqueeze(1).expand(B, N, N, dm)
            kap = self.k_mlp(torch.cat([hi, hj], -1)).squeeze(-1)
            pull = torch.einsum('bij,bjd->bid', self.K.to(z.dtype).unsqueeze(0) * kap, z)
            vel = vel + tangent(z, pull)
        if cfg.drive == 'stimulus':
            s = self.stim(h)
            if cfg.stim_ema > 0:
                self._stim_state = s if self._stim_state is None else cfg.stim_ema * self._stim_state + (1 - cfg.stim_ema) * s
                s = self._stim_state
            vel = vel + tangent(z, s)
        if cfg.phase_step_max_deg > 0:
            cap = math.tan(math.radians(cfg.phase_step_max_deg))
            step_norm = (vel * cfg.dt).norm(dim=-1, keepdim=True)
            vel = vel * torch.clamp(cap / (step_norm + 1e-8), max=1.0)
        return sphere_step(z, vel, cfg.dt)

    def _receive_vector(self, h: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.cfg
        B, N, _ = h.shape
        m = self.msg_proj(h)                                               # (B, N, D)
        if self.n_heads and not cfg.head_transmits:
            m = torch.cat([m[:, :self.M], torch.zeros_like(m[:, self.M:])], 1)
        eye = torch.eye(N, device=h.device, dtype=h.dtype)
        g = torch.einsum('bid,bjd->bij', z, z)
        if cfg.echo_cancel:
            g = g * (1 - eye)
        if cfg.medium == 'channels':
            return self._receive(h, torch.zeros(B, N, device=h.device, dtype=h.dtype))[0], g
        if cfg.bus_phase == 'zero':
            return torch.zeros(B, N, m.shape[-1] * self.n_rx, device=h.device, dtype=h.dtype), \
                torch.zeros_like(g)
        bus = torch.einsum('bnD,bnd->bDd', m, z)                            # (B, D, d), shared
        Fr = self._frame(z)                                                # (B, N, n_rx, d)
        r = torch.einsum('bDd,bnad->bnDa', bus, Fr)                        # (B, N, D, n_rx)
        if cfg.echo_cancel:
            r = r - torch.einsum('bnD,bnd,bnad->bnDa', m, z, Fr)
        return r.flatten(2) / float(N), g

    # ---------------- scalar phases (original) ----------------

    def _init_theta(self, B: int, dev, dtype) -> Tensor:
        M = self.N
        if self.cfg.theta_init == 'zero':
            return torch.zeros(B, M, device=dev, dtype=dtype)
        if self.cfg.theta_init == 'learned':
            return self.theta0.unsqueeze(0).expand(B, M).to(dtype)
        return 2 * torch.pi * torch.rand(B, M, device=dev, dtype=dtype)

    def _phase_step(self, theta: Tensor, h: Tensor) -> Tensor:
        cfg = self.cfg
        vel = self.omega.to(theta.dtype).expand_as(theta)
        if cfg.coupling == 'mlp':
            B, M, dm = h.shape
            hi = h.unsqueeze(2).expand(B, M, M, dm); hj = h.unsqueeze(1).expand(B, M, M, dm)
            kap = self.k_mlp(torch.cat([hi, hj], -1)).squeeze(-1)
            diff = theta.unsqueeze(1) - theta.unsqueeze(2)                # theta_j - theta_i
            vel = vel + (self.K.to(theta.dtype) * torch.sin(diff) * kap).sum(-1)
        if cfg.drive == 'stimulus':
            c = self.stim(h)
            vel = vel - c[..., 0] * torch.sin(theta) + c[..., 1] * torch.cos(theta)
        return theta + cfg.dt * vel

    def _receive(self, h: Tensor, theta: Tensor) -> tuple[Tensor, Tensor]:
        """-> received signal (B, M, rx), effective gate (B, M, M) for metrics."""
        cfg = self.cfg
        B, M, _ = h.shape
        m = self.msg_proj(h)                                               # (B, N, D)
        if self.n_heads and not cfg.head_transmits:
            m = torch.cat([m[:, :self.M], torch.zeros_like(m[:, self.M:])], 1)
        if cfg.medium == 'channels':
            if cfg.gate_mode == 'full':
                Nn = h.shape[1]
                g = torch.ones(B, Nn, Nn, device=h.device, dtype=h.dtype)
                if cfg.echo_cancel:
                    g = g * (1 - torch.eye(Nn, device=h.device, dtype=h.dtype))
                # receiver i sees [g_i1 m_1; ...; g_iN m_N]: full per-sender access
                r = torch.einsum('bij,bjk->bijk', g, m).reshape(B, Nn, Nn * m.shape[-1])
                return r, g
            if cfg.gate_mode == 'attn':
                gq, gk = self.gate_q(h), self.gate_k(h)
                g = F.softmax(torch.einsum('bik,bjk->bij', gq, gk) / gq.shape[-1] ** 0.5, -1)
                norm = 1.0
            elif cfg.gate_mode == 'open':
                g = torch.ones(B, M, M, device=h.device, dtype=h.dtype); norm = float(M)
            else:
                g = torch.zeros(B, M, M, device=h.device, dtype=h.dtype); norm = float(M)
            if cfg.echo_cancel:
                g = g * (1 - torch.eye(M, device=h.device, dtype=h.dtype))
            return torch.einsum('bij,bjk->bik', g, m) / norm, g
        # bus
        if cfg.bus_phase == 'zero':
            g = torch.zeros(B, M, M, device=h.device, dtype=h.dtype)
            return torch.zeros(B, M, m.shape[-1] * (2 if cfg.quadrature else 1),
                               device=h.device, dtype=h.dtype), g
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)                  # (B, M)
        bus_re = (m * cos_t.unsqueeze(-1)).sum(1, keepdim=True)            # (B, 1, D)
        bus_im = (m * sin_t.unsqueeze(-1)).sum(1, keepdim=True)
        if cfg.echo_cancel:
            bus_re = bus_re - m * cos_t.unsqueeze(-1)
            bus_im = bus_im - m * sin_t.unsqueeze(-1)
        r_i = bus_re * cos_t.unsqueeze(-1) + bus_im * sin_t.unsqueeze(-1)  # sum_j m_j cos(th_j - th_i)
        r = r_i / float(M)
        if cfg.quadrature:
            r_q = (-bus_re * sin_t.unsqueeze(-1) + bus_im * cos_t.unsqueeze(-1)) / float(M)
            r = torch.cat([r, r_q], -1)
        g = torch.cos(theta.unsqueeze(-1) - theta.unsqueeze(-2))           # effective gate
        if cfg.echo_cancel:
            g = g * (1 - torch.eye(M, device=h.device, dtype=h.dtype))
        return r, g

    def _update(self, inp: Tensor, h: Tensor) -> Tensor:
        B, M, dm = h.shape
        if self.cfg.per_module_gru:
            return torch.stack([self.cells[k](inp[:, k], h[:, k]) for k in range(M)], 1)
        return self.cell(inp.reshape(B * M, -1), h.reshape(B * M, dm)).reshape(B, M, dm)

    # ------------------------------------------------------------------

    def forward(self, images: Tensor, questions: Tensor, t_override: int | None = None,
                phase_override: str | None = None, return_trace: bool = False, **batch) -> dict:
        cfg = self.cfg
        B, M, Nq = images.shape[0], self.M, self.Nq
        T = t_override if t_override is not None else self.T
        dev = images.device
        questions = questions.float()                                       # the task stores them as ints
        squeeze = questions.dim() == 2
        if squeeze:
            questions = questions.unsqueeze(1)                              # (B, 1, 18)
        q_all = questions.reshape(B, -1)                                    # (B, Nq*18)

        h = self.h_init(q_all).reshape(B, M, self.dm)
        if not (self.slots or self.field_fe):
            h = h + self.module_embed.unsqueeze(0)
        read_attn = None
        anchors = None
        tok_phase = None
        if self.field_fe:
            f = self.field_enc(images)
            f = f * (1 + self.grid_film_gamma(q_all))[..., None, None] + self.grid_film_beta(q_all)[..., None, None]
            f = self.grid_norm(f)
            if cfg.use_pos_emb:
                f = f + self.pos_emb
            zf = self.field(f, freeze=phase_override == 'freeze_field')
            Zt = self.field.to_tokens(zf)                                   # (B, P, K_f, d_f)
            feats = f.flatten(2).transpose(1, 2)                            # (B, P, fch)
            X, anchors, read_attn = self.phase_slots(feats, Zt)             # (B, M, tok), (B, M, K_f, d_f), (B, M, P)
            tok_phase = F.normalize(Zt.flatten(2), dim=-1)                  # (B, P, K_f*d_f) for the analysis callback
        elif self.slots:
            f = self.encoder(images)
            f = f * (1 + self.grid_film_gamma(q_all))[..., None, None] + self.grid_film_beta(q_all)[..., None, None]
            f = self.grid_norm(f)
            if cfg.use_pos_emb:
                f = f + self.pos_emb
            X, read_attn = self.slot_attn(f.flatten(2).transpose(1, 2))    # (B, M, tok), (B, M, P)
        elif self.objects:
            feats, found = self.obj_tok(images)
            self._last_found = found
            X = self.obj_embed(feats)                                       # (B, M, tok)
        else:
            # colour-keyed slots: module k reads its object from the grid
            f = self.encoder(images)
            f = f * (1 + self.grid_film_gamma(q_all))[..., None, None] + self.grid_film_beta(q_all)[..., None, None]
            f = self.grid_norm(f)
            if cfg.use_pos_emb:
                f = f + self.pos_emb
            grid_tokens = f.flatten(2).transpose(1, 2)                      # (B, P, ch)
            grid_keys = F.normalize(grid_tokens, dim=-1)
            X, read_attn = self._grid_read(h, grid_tokens, grid_keys)       # (B, M, tok), (B, M, P)
        X = self.norm(X * (1 + self.film_gamma(q_all)).unsqueeze(1) + self.film_beta(q_all).unsqueeze(1))
        reread = (not self.objects) and (not self.slots) and (not self.field_fe) and cfg.read_every_step
        if self.n_heads:
            h_heads = self.head_init(questions).repeat_interleave(self.taps, 1) + self.head_embed.unsqueeze(0)  # (B, Nq*taps, dm)
            h = torch.cat([h, h_heads], 1)
            X = torch.cat([X, torch.zeros(B, self.n_heads, X.shape[-1], device=dev, dtype=X.dtype)], 1)
        self._stim_state = None
        if self.vector:
            z = self._init_z(B, dev, X.dtype)
            if (cfg.medium == 'bus' and cfg.bus_phase == 'open') or phase_override == 'zero':
                z = torch.zeros_like(z); z[..., 0] = 1.0
            elif phase_override == 'shuffle':
                perm = torch.argsort(torch.rand(B, self.N, device=dev), dim=1)
                z = z.gather(1, perm.unsqueeze(-1).expand_as(z))
            theta = None
        else:
            theta = self._init_theta(B, dev, X.dtype)
            if cfg.medium == 'bus' and cfg.bus_phase == 'open':
                theta = torch.zeros_like(theta)
            if phase_override == 'zero':
                theta = torch.zeros_like(theta)
            elif phase_override == 'shuffle':
                perm = torch.argsort(torch.rand(B, self.N, device=dev), dim=1)
                theta = theta.gather(1, perm)
        if anchors is not None and cfg.slot_phase_from_anchor and cfg.medium == 'bus' and cfg.bus_phase == 'phase' \
                and phase_override not in ('zero', 'shuffle'):
            a = self.anchor_to_phase(anchors.flatten(2))                    # (B, M, d|2)
            if self.vector:
                z = torch.cat([F.normalize(a, dim=-1), z[:, M:]], 1)
            else:
                theta = torch.cat([torch.atan2(a[..., 1], a[..., 0]), theta[:, M:]], 1)
        evolve = (cfg.medium == 'bus' and cfg.bus_phase == 'phase'
                  and phase_override not in ('zero', 'freeze'))

        traces = {'phase': [], 'gates': [], 'h': [], 'attn': []} if return_trace else None
        if return_trace and tok_phase is not None:
            traces['tok_phase'] = [tok_phase.detach()]
        g = torch.zeros(B, self.N, self.N, device=dev, dtype=X.dtype)
        g_hist: list[Tensor] = []
        # read trace for the analysis callbacks: object rows read their own
        # object (identity) or their learned grid attention; head rows read nothing
        if read_attn is None:
            attn_fixed = torch.cat([torch.eye(M, device=dev, dtype=X.dtype),
                                    torch.full((self.n_heads, M), 1.0 / M, device=dev, dtype=X.dtype)], 0
                                   ).unsqueeze(0).expand(B, self.N, M)
        else:
            P = read_attn.shape[-1]
            attn_fixed = torch.cat([read_attn, torch.full((B, self.n_heads, P), 1.0 / P, device=dev, dtype=X.dtype)], 1)
        for _ in range(T):
            if self.vector:
                r, g = self._receive_vector(h, z)
            else:
                r, g = self._receive(h, theta)
            g_hist.append(g)
            if reread:
                Xm, read_attn = self._grid_read(h[:, :M], grid_tokens, grid_keys)
                Xm = self.norm(Xm * (1 + self.film_gamma(q_all)).unsqueeze(1) + self.film_beta(q_all).unsqueeze(1))
                X = torch.cat([Xm, torch.zeros(B, self.n_heads, Xm.shape[-1], device=dev, dtype=Xm.dtype)], 1)
                P = read_attn.shape[-1]
                attn_fixed = torch.cat([read_attn, torch.full((B, self.n_heads, P), 1.0 / P, device=dev, dtype=X.dtype)], 1)
            h = self._update(torch.cat([X, r], -1), h)
            if evolve:
                if self.vector:
                    z = self._z_step(z, h)
                else:
                    theta = self._phase_step(theta, h)
            if return_trace:
                zt = z if self.vector else torch.stack([torch.cos(theta), torch.sin(theta)], -1)
                traces['phase'].append(zt.detach())                                                      # type: ignore
                traces['gates'].append(g.detach()); traces['h'].append(h.detach())                       # type: ignore
                traces['attn'].append(attn_fixed)                                                        # type: ignore

        if cfg.readout == 'head':
            h_tap = h[:, M:].reshape(B, Nq, self.taps * self.dm)                  # taps concatenated per question
            logits = self.head_out(torch.cat([h_tap, questions], -1))             # (B, Nq, A)
        elif cfg.readout == 'asker':
            n_obj = self.obj_tok.n_objects
            asker = questions[..., :n_obj].argmax(-1)                                  # (B, Nq)
            h_ask = h.gather(1, asker.unsqueeze(-1).expand(B, Nq, self.dm))            # (B, Nq, dm)
            logits = self.asker_head(torch.cat([h_ask, questions], -1))
        else:
            ho = h[:, :M]
            hq = torch.cat([ho.unsqueeze(2).expand(B, M, Nq, self.dm),
                            questions.unsqueeze(1).expand(B, M, Nq, cfg.q_size)], -1)
            logits = torch.stack([self.vote_heads[k](hq[:, k]) for k in range(M)], 1).sum(1)
        if cfg.use_prior_head:
            logits = logits + self.prior_head(questions)
        if squeeze:
            logits = logits.squeeze(1)

        with torch.no_grad():
            n = self.obj_tok.n_objects
            named = ((questions[..., :n] + questions[..., n:2 * n]) > 0).float()   # (B, Nq, M)
            zu = z if self.vector else torch.stack([torch.cos(theta), torch.sin(theta)], -1)   # (B, N, d)
            zo = zu[:, :M]
            if named.shape[-1] == M:                       # module index <-> object identity (objects / colour keys)
                same = torch.einsum('bqi,bqj->bij', named, named) > 0              # same question
                cross = (torch.einsum('bqi,brj->bij', named, named) > 0) & ~same   # different questions
            else:                                          # exchangeable slots, M != n_objects: no identity map
                same = torch.zeros(zo.shape[0], M, M, dtype=torch.bool, device=dev)
                cross = same
            off = ~torch.eye(M, dtype=torch.bool, device=dev)
            cosm = torch.einsum('bid,bjd->bij', zo, zo)
            ev = torch.linalg.eigvalsh(0.5 * (1 + cosm).float()).clamp(min=0)
            metrics = {
                'phase_R': zo.mean(1).norm(dim=-1).mean().item(),
                'gate_offdiag': g[:, :M, :M][:, off].mean().item(),
                'n_clusters_eff': ((ev.sum(-1) ** 2) / ((ev ** 2).sum(-1) + 1e-6)).mean().item(),
            }
            if self.objects:
                metrics['obj_found'] = self._last_found.float().sum(-1).mean().item()
            elif read_attn is not None:
                an = F.normalize(read_attn, dim=-1)
                metrics['read_overlap'] = torch.einsum('bmp,bnp->bmn', an, an)[:, off].mean().item()
                ent = -(read_attn.clamp(min=1e-12) * read_attn.clamp(min=1e-12).log()).sum(-1)
                metrics['read_entropy'] = (ent / float(np.log(read_attn.shape[-1]))).mean().item()
            if (same & off).any():
                metrics['same_q_align'] = cosm[same & off].mean().item()
            if (cross & off).any():
                metrics['cross_q_align'] = cosm[cross & off].mean().item()
                metrics['coalition_score'] = metrics['same_q_align'] - metrics['cross_q_align']
            if len(g_hist) > 1:
                gs = torch.stack(g_hist)                                              # (T, B, N, N)
                metrics['gate_tvar'] = gs[:, :, :M, :M].var(0, unbiased=False)[:, off].mean().item()
                if self.n_heads:
                    metrics['head_tvar'] = gs[:, :, M:, :M].var(0, unbiased=False).mean().item()
            if self.n_heads and named.shape[-1] == M and self.taps == 1:
                # is each head in phase with the pair it asks about, and out of phase with the other pair?
                ch = torch.einsum('bqd,bmd->bqm', zu[:, M:], zo)                      # (B, Nq, M)
                own = (ch * named).sum(-1) / named.sum(-1).clamp(min=1)
                if Nq > 1:
                    other = named.sum(1, keepdim=True) - named                         # modules named by other questions
                else:
                    other = 1.0 - named                                               # modules the question does not name
                oth = (ch * other).sum(-1) / other.sum(-1).clamp(min=1)
                metrics['head_own_align'] = own.mean().item()
                metrics['head_other_align'] = oth.mean().item()
        return {'logits': logits, 'traces': traces, 'metrics': metrics}
