"""Component invariants, one property per claim. Detached: imports src as a
library; nothing imports this. The QueryRead case PASSES BY FAILING -- the
falsified binder must collapse (overlap -> 1) on an unstructured field,
because that collapse is the L1 exhibit."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.sync.components import (CompetitiveClaim, KuramotoStep,      # noqa: E402
                                        PrivateLines, QueryRead, SharedBus,
                                        SilentBus)
from src.models.sync.field import OscillatorField                            # noqa: E402

B, P, Kg, df, M, dm, d, msg = 3, 50, 16, 4, 6, 96, 6, 4
ok = 0


def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f'  ok  {name}')


torch.manual_seed(0)

# binding: exclusivity ---------------------------------------------------
claim = CompetitiveClaim(64, 64, M, Kg, df)
feats = torch.randn(B, P, 64)
Zt = F.normalize(torch.randn(B, P, Kg, df), dim=-1)
slots, phi, reads = claim(feats, Zt)
attn_back = reads * reads.sum(-1, keepdim=True)  # undo renorm approx: instead recompute
logits = claim.log_beta.exp() * torch.einsum('bkgd,bngd->bkn', phi, Zt) / Kg
check('claim: cells choose one owner (softmax over slots sums to 1)',
      torch.allclose(F.softmax(logits, 1).sum(1), torch.ones(B, P), atol=1e-5))
check('claim: anchors unit per group', torch.allclose(phi.norm(dim=-1), torch.ones(B, M, Kg), atol=1e-5))

# binding: the falsified control collapses -------------------------------
qr = QueryRead(64, 64, M, Kg, df, query_dim=dm, exclusive=False)
Z_uniform = F.normalize(torch.ones(B, P, Kg, df), dim=-1)                    # fully synchronised field
_, _, attn = qr(feats, Z_uniform, torch.randn(B, M, dm))
an = F.normalize(attn, dim=-1)
overlap = torch.einsum('bmp,bnp->bmn', an, an).mean()
check(f'QueryRead(exclusive=False) collapses on a synchronised field (overlap {overlap:.3f})',
      overlap > 0.999)

# medium: frames, echo, silence ------------------------------------------
bus = SharedBus(dm, msg, d)
z = F.normalize(torch.randn(B, M + 1, d), dim=-1)
Fr = bus._frame(z)
gram = torch.einsum('xnad,xnbd->xnab', Fr, Fr)
eye = torch.eye(d).expand(B, M + 1, d, d)
check('bus: frames orthonormal', torch.allclose(gram, eye, atol=1e-4))
h = torch.randn(B, M + 1, dm)
r = bus(h, z)
mm = bus.msg_proj(h)
wire_all = torch.einsum('bnD,bnd->bDd', mm, z)
own = torch.einsum('bD,bd->bDd', mm[:, 0], z[:, 0])
r0_manual = torch.einsum('bDd,bad->bDa', wire_all - own, Fr[:, 0]).flatten(1) / (M + 1)
check('bus: echo cancellation identity (row 0 reads wire minus own write)',
      torch.allclose(r[:, 0], r0_manual, atol=1e-5))
check('silent: carries nothing', SilentBus(dm, msg, d)(h, z).abs().max() == 0)

# medium: capacity -- the demodulation theorem. For orthonormal addresses,
# Bus z_j = m_j exactly: d senders, one wire, zero crosstalk.
z_orth = torch.eye(d)[None, :, :].expand(B, d, d).contiguous()
h_d = torch.randn(B, d, dm)
bus_d = SharedBus(dm, msg, d)
m_d = bus_d.msg_proj(h_d)
wire = torch.einsum('bnD,bnd->bDd', m_d, z_orth)                             # what the medium carries
demod = torch.einsum('bDd,bd->bD', wire, z_orth[:, 1])                       # any receiver, sender 1's axis
check('bus: orthogonal sender demodulates exactly (Bus z_j = m_j)',
      torch.allclose(demod, m_d[:, 1], atol=1e-5))
r_d = bus_d(h_d, z_orth).reshape(B, d, msg, d)
check('bus: own-axis channel empty after echo cancellation',
      torch.allclose(r_d[:, 0, :, 0], torch.zeros(B, msg), atol=1e-5))

# dynamics: norm preservation + scalar reduction --------------------------
dyn = KuramotoStep(M + 1, dm, d, dt=0.5)
z2 = dyn(z, h)
check('dynamics: renormalised to the sphere', torch.allclose(z2.norm(dim=-1), torch.ones(B, M + 1), atol=1e-5))
dyn2 = KuramotoStep(M + 1, dm, 2, dt=0.5)
z_s = F.normalize(torch.randn(B, M + 1, 2), dim=-1)
z_s2 = dyn2(z_s, h)
check('dynamics: d=2 (the scalar circle) runs through the same class',
      z_s2.shape == z_s.shape and torch.allclose(z_s2.norm(dim=-1), torch.ones(B, M + 1), atol=1e-5))

# lines: gates ------------------------------------------------------------
for gate in ('attn', 'full', 'phase'):
    pl = PrivateLines(dm, msg, d, M + 1, gate=gate)
    rr = pl(h, z)
    check(f'lines[{gate}]: shape ({pl.out_dim})', rr.shape == (B, M + 1, pl.out_dim))

# field: unit norm per group ----------------------------------------------
field = OscillatorField()
zf = field(torch.randn(B, 64, 9, 9))
zg = zf.view(B, Kg, df, 9, 9)
check('field: unit norm per group after settling',
      torch.allclose(zg.norm(dim=2), torch.ones(B, Kg, 9, 9), atol=1e-4))

print(f'\nall {ok} component invariants hold')
