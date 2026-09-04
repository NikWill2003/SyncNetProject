#!/usr/bin/env python3
"""Post-hoc intervention suite on saved checkpoints -- for SQOOP, where the
intervention callback does not run in training (it is Sort-of-CLEVR-scoped).

Answers the question the gated SQOOP curve leaves open: is the phase gate
LOAD-BEARING there, or inert as on Sort-of-CLEVR? For each run directory
(a hydra output dir holding .hydra/config.yaml and best_model.pt) it rebuilds
the model, loads the best weights, and evaluates the test split under every
override the model declares:

    phase: freeze | zero | shuffle | anchor_shuffle   (PHASE_OVERRIDES)
    gate : open | zero                                (GATE_OVERRIDES)

    python scripts/run_interventions.py outputs/sqoop/2026-09-03/*/
    python scripts/run_interventions.py --glob 'outputs/sqoop/**/best_model.pt' --csv results/sqoop_interventions.csv

Prints baseline accuracy and the DROP under each intervention; a drop near
zero means the model does not use that variable.
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.registry import build_dataloaders, build_model, register_configs  # noqa: E402


def load_run(run_dir: Path, device: str):
    cfg = OmegaConf.load(run_dir / '.hydra' / 'config.yaml')
    cfg.wandb.enabled = False
    cfg.train.compile_model = False
    model = build_model(cfg).to(device)
    state = torch.load(run_dir / 'best_model.pt', map_location=device)
    if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
        state = state['model']
    clean = {}
    for k, v in state.items():
        for pre in ('_orig_mod.', 'module.'):
            if k.startswith(pre):
                k = k[len(pre):]
        clean[k] = v
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing or unexpected:
        print(f'  (state dict: {len(missing)} missing, {len(unexpected)} unexpected keys)')
    model.eval()
    return cfg, model


@torch.inference_mode()
def accuracy(model, loader, device, **overrides) -> float:
    correct = total = 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch, **overrides)
        logits = out['logits'] if isinstance(out, dict) else out
        correct += int((logits.argmax(-1) == batch['answers']).sum())
        total += int(batch['answers'].numel())
    return correct / max(total, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*', help='hydra run directories')
    ap.add_argument('--glob', default=None, help="e.g. 'outputs/sqoop/**/best_model.pt'")
    ap.add_argument('--csv', default=None)
    ap.add_argument('--split', default='test', choices=['eval', 'test'])
    ap.add_argument('--t-ramp', default=None, metavar='T,T,...',
                    help="also evaluate at these internal step counts (test-time T ramp), "
                         "e.g. 1,2,4,6,8,12,16")
    args = ap.parse_args()

    run_dirs = [Path(r) for r in args.runs]
    if args.glob:
        run_dirs += [Path(p).parent for p in glob.glob(args.glob, recursive=True)]
    run_dirs = [r for r in dict.fromkeys(run_dirs) if (r / 'best_model.pt').is_file()]
    if not run_dirs:
        raise SystemExit('no run directories with best_model.pt found')

    register_configs()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = []
    for rd in run_dirs:
        cfg, model = load_run(rd, device)
        loaders = build_dataloaders(cfg, device)
        loader = loaders[2 if args.split == 'test' else 1]
        name = f"{cfg.model.name} rhs={cfg.dataset.get('rhs_variety', '-')} seed={cfg.train.seed}"
        base = accuracy(model, loader, device)
        row = {'run': str(rd), 'model': cfg.model.name, 'rhs': cfg.dataset.get('rhs_variety', ''),
               'seed': cfg.train.seed, 'baseline': round(base, 4)}
        print(f'\n{name}   baseline {base:.4f}   [{rd}]')
        for p in getattr(model, 'PHASE_OVERRIDES', ()):
            a = accuracy(model, loader, device, phase_override=p)
            row[f'phase_{p}_drop'] = round(base - a, 4)
            print(f'   phase {p:14s} {a:.4f}   drop {base - a:+.4f}')
        for g in getattr(model, 'GATE_OVERRIDES', ()):
            a = accuracy(model, loader, device, gate_override=g)
            row[f'gate_{g}_drop'] = round(base - a, 4)
            print(f'   gate  {g:14s} {a:.4f}   drop {base - a:+.4f}')
        if args.t_ramp:
            for t in [int(x) for x in args.t_ramp.split(',')]:
                a = accuracy(model, loader, device, t_override=t)
                row[f't_ramp_T{t}'] = round(a, 4)
                print(f'   T={t:<3}               {a:.4f}   ({a - base:+.4f} vs trained T)')
        rows.append(row)

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('run', 'model', 'rhs', 'seed', 'baseline'), k))
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f'\nwrote {args.csv} ({len(rows)} runs)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
