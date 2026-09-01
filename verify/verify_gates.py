"""By-hand runner for the registered-prediction gates on synthetic data.
Small and CPU-honest: --steps controls cost. The synthetic gate certifies
the GRADIENT PATH (can the architecture fit arbitrary labels on inputs it
can tell apart?); the perception-collapse half of the SQOOP prediction
needs the real stimulus-starved images, so run this on the node against
real batches for the registered comparison. Detached: imports src as a
library; nothing imports this."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.probes import grad_decomposition, memorisation_gate       # noqa: E402
from src.models.sync.busnet import BusNet, BusNetConfig                     # noqa: E402
from src.models.sync.gated import GatedNet, GatedNetConfig                  # noqa: E402
from src.models.sync.token_busnet import TokenBusNet, TokenBusNetConfig     # noqa: E402


def synth_batch(B: int, seed: int = 11):
    g = torch.Generator().manual_seed(seed)
    return {'images': torch.rand(B, 3, 64, 64, generator=g),
            'questions': torch.randint(0, 36, (B, 3), generator=g),
            'scenes': torch.cat([torch.randint(0, 36, (B, 5, 1), generator=g),
                                 torch.randint(0, 64, (B, 5, 2), generator=g)], -1),
            'answers': torch.zeros(B, dtype=torch.long)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()
    b = synth_batch(args.batch)

    for name, model in (
            ('token_busnet', TokenBusNet(TokenBusNetConfig(q_onehot_vocab=40), 'sqoop', 2)),
            ('gated', GatedNet(GatedNetConfig(q_onehot_vocab=40), 'sqoop', 2)),
            ('busnet', BusNet(BusNetConfig(q_onehot_vocab=40), 'sqoop', 2))):
        torch.manual_seed(0)
        res = memorisation_gate(model, b, n_classes=2, steps=args.steps)
        print(f'{name:14s} gate: first {res["acc_first"]:.2f} -> last {res["acc_last"]:.2f} '
              f'({"PASS" if res["passes"] else "fail"} at {args.steps} steps)')
        gd = grad_decomposition(model, b)
        top = ', '.join(f'{k} {v:.2f}' for k, v in sorted(gd.items(), key=lambda kv: -kv[1])[:3])
        print(f'{"":14s} grad: {top}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
