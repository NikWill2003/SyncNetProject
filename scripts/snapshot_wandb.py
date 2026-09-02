"""Snapshot a wandb project through the API into the repo -- the durable,
untruncated record the CSV export button is not (it drops newer columns:
anchor_shuffle_* and the fixed-point signature both vanished from exports
while existing in the run summaries).

  python scripts/snapshot_wandb.py --entity <you> --project sort_of_clevr
  python scripts/snapshot_wandb.py ... --tag explore1
  python scripts/snapshot_wandb.py ... --history callbacks/read_overlap,callbacks/field_R

Writes results/<project>_snapshot_<ts>.csv (every summary + config key of
every matched run) and, with --history, results/<project>_history_<ts>.csv
in long format (run, name, step, key, value; ~500 samples/run) -- the file
the early-predictor scatter (read_overlap@5k vs final accuracy) reads.
Plain argparse, no hydra. Needs WANDB_API_KEY."""

import argparse
import time
from pathlib import Path

import pandas as pd
import wandb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--entity', required=True)
    ap.add_argument('--project', required=True)
    ap.add_argument('--tag', default=None, help='only runs carrying this tag')
    ap.add_argument('--history', default=None,
                    help='comma-separated metric keys to pull per-step')
    ap.add_argument('--samples', type=int, default=500)
    args = ap.parse_args()

    api = wandb.Api(timeout=60)
    filters = {'tags': args.tag} if args.tag else {}
    runs = list(api.runs(f'{args.entity}/{args.project}', filters=filters))
    print(f'{len(runs)} runs matched')

    ts = time.strftime('%Y%m%d_%H%M')
    out = Path('results')
    out.mkdir(exist_ok=True)

    rows = []
    for r in runs:
        row = {'run_id': r.id, 'name': r.name, 'state': r.state,
               'created': str(r.created_at), 'tags': ','.join(r.tags)}
        row.update({f'config/{k}': v for k, v in r.config.items()
                    if not k.startswith('_')})
        row.update({k: v for k, v in r.summary.items()
                    if isinstance(v, (int, float, str, bool))})
        rows.append(row)
    snap = out / f'{args.project}_snapshot_{ts}.csv'
    pd.DataFrame(rows).to_csv(snap, index=False)
    print(f'wrote {snap} ({len(rows)} rows, every key)')

    if args.history:
        keys = [k.strip() for k in args.history.split(',')]
        hrows = []
        for r in runs:
            for rec in r.history(keys=keys + ['_step'],
                                 samples=args.samples, pandas=False):
                for k in keys:
                    if rec.get(k) is not None:
                        hrows.append({'run_id': r.id, 'name': r.name,
                                      'step': rec.get('_step'),
                                      'key': k, 'value': rec[k]})
        hist = out / f'{args.project}_history_{ts}.csv'
        pd.DataFrame(hrows).to_csv(hist, index=False)
        print(f'wrote {hist} ({len(hrows)} points)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
