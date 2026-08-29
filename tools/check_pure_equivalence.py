"""Verify PureSyncNet == the canonical BusNet cell, weight for weight."""
import sys, torch
sys.path.insert(0, '.')
from src.models.busnet import BusNet, BusNetConfig
from src.models.syncnet_pure import PureSyncNet
from src.tasks.sort_of_clevr.data import constants as C

CAN = dict(name='b', encoder={'name': 'field'}, per_module_gru=False,
           phase_repr='vector', osc_dim=6, drive='stimulus')
torch.manual_seed(0)
full = BusNet(BusNetConfig(**CAN), 75, 10, list(C.COLOURS.values()))
pure = PureSyncNet(75, 10)
ps = pure.state_dict()
pure.load_state_dict({k: v for k, v in full.state_dict().items() if k in ps}, strict=True)
x = torch.randn(8, 3, 75, 75)
q = torch.zeros(8, 18); q[:, 0] = 1; q[:, 6] = 1
with torch.no_grad():
    torch.manual_seed(1); a = full(x, q)['logits']
    torch.manual_seed(1); b = pure(x, q)['logits']
d = (a - b).abs().max().item()
print(f'params {sum(p.numel() for p in pure.parameters())}  max|dlogit| {d}')
assert d == 0.0, 'pure model diverges from the canonical cell'
print('OK: PureSyncNet is the canonical cell, bit for bit')
