import sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
from src.models.question_encoders import IdentityQuestionEncoder
from src.models.syncnet import VQASyncNet, VQASyncNetConfig
from syncnet_orig import VQASyncNet as OrigNet, VQASyncNetConfig as OrigCfg

def run(cls, cfgcls, seed, **kw):
    torch.manual_seed(seed)
    cfg = cfgcls(name='x', **kw)
    m = cls(cfg, IdentityQuestionEncoder(18), 75, 10).eval()
    torch.manual_seed(123)
    imgs = torch.rand(8, 3, 75, 75); qs = torch.zeros(8, 18); qs[:, 0] = 1; qs[:, 7] = 1; qs[:, 13] = 1; qs[:, 16] = 1
    torch.manual_seed(7)
    with torch.no_grad():
        out = m(imgs, qs)
    return out['logits'], out['metrics'], sum(p.numel() for p in m.parameters())

for kw in [dict(), dict(partition='quadrant', readout_mode='sum', q_conditioning='film'),
           dict(partition='quadrant', readout_mode='sum', gate_mode='attn'),
           dict(partition='quadrant', readout_mode='sum', gate_mode='frozen'),
           dict(gate_mode='mlp', readout_mode='both'),
           dict(readout_mode='sync', deterministic_phase=True), dict(msg_agg='bus', gate_mode='open'),
           dict(partition='views', n_modules=3, readout_mode='sum')]:
    lo, mo, no = run(OrigNet, OrigCfg, 0, **kw)
    ln, mn, nn_ = run(VQASyncNet, VQASyncNetConfig, 0, **kw)
    print(kw, '\n   params', no, nn_, 'max|dlogit|', (lo - ln).abs().max().item(),
          'R', round(mo['phase_R'],4), round(mn['phase_R'],4), 'off', round(mo['gate_offdiag'],4), round(mn['gate_offdiag'],4))
