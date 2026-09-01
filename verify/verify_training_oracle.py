"""Training-loop equivalence, beyond forward equivalence: the new
composition, started from the SAME golden weights and fed the SAME batch
and RNG stream, must reproduce the old implementation's 40-step AdamW loss
curve and final logits. Weight decay is ON (0.01) deliberately: the one
parameter the old model carries and the new one does not (the dead
module_embed) receives zero gradient and decoupled decay only, so shared
trajectories must still match exactly -- this script is the proof.
Detached: imports src as a library; nothing imports this."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.sync.busnet import BusNet, BusNetConfig                      # noqa: E402
from verify.verify_reference import map_key                                  # noqa: E402


def batch(t: int):
    g = torch.Generator().manual_seed(1000 + t)
    x = torch.rand(8, 3, 75, 75, generator=g)
    q = torch.zeros(8, 18)
    q[torch.arange(8), torch.randint(0, 18, (8,), generator=g)] = 1
    y = torch.randint(0, 10, (8,), generator=g)
    return {'images': x, 'questions': q, 'answers': y}


def main() -> int:
    fx = ROOT / 'verify' / 'fixtures'
    gold = torch.load(fx / 'canonical_golden.pt', weights_only=False)
    oracle = torch.load(fx / 'oracle_canonical_curve.pt', weights_only=False)

    model = BusNet(BusNetConfig(), 'sort_of_clevr', 10)
    mapped = {}
    for k, v in gold['state_dict'].items():
        nk = map_key(k.removeprefix('inner.'))
        if nk:
            mapped[nk] = v
    model.load_state_dict(mapped, strict=True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=oracle['lr'], weight_decay=oracle['wd'])

    losses = []
    for t in range(int(oracle['n_steps'])):
        b = batch(t)
        torch.manual_seed(5000 + t)
        out = model(b)
        loss = F.cross_entropy(out['logits'], b['answers'])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    dl = (torch.tensor(losses, dtype=torch.float64) - oracle['losses']).abs().max().item()
    model.eval()
    with torch.no_grad():
        torch.manual_seed(123)
        fin = model({'images': gold['images'], 'questions': gold['questions'],
                     'answers': torch.zeros(6, dtype=torch.long)})['logits']
    dg = (fin - oracle['final_logits']).abs().max().item()
    print(f'max |loss diff| over {int(oracle["n_steps"])} steps = {dl}')
    print(f'max |final logit diff| after training       = {dg}')
    assert dl < 1e-6 and dg < 1e-5, 'training trajectories diverged'
    print('TRAINING-EQUIVALENT: same weights, same stream, same trajectory.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
