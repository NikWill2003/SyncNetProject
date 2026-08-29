"""Plateau + data diagnostics for the SQOOP BusNet.

Two questions, one script:

1. Is the tiny gradient CANCELLATION (healthy per-example signal, opposed
   across classes) or a genuinely dead network?
     - batch grad norm at several batch sizes: pure cancellation scales like
       1/sqrt(B); a dead network is tiny and B-independent.
     - half-batch gradient cosine: ~0 on a signal-free plateau, > 0 once a
       real signal direction exists.
     - per-example grad norm: healthy magnitudes rule out saturation.

2. Is the DATA sound?
     - answer balance per split; per-question (x, rel, y) both-answer balance
     - the SQOOP contract: unseen-split question triples never occur in train
     - rhs_variety: distinct rhs per lhs in train matches the config
     - image sanity: dtype/range, non-empty glyph pixels, no duplicate rows

Usage (on the node, real rhs=18 dir):
    python tools/diagnose_sqoop.py --root ./data \
        --dir sqoop-seed0-train1080000-test25600-rhs18-img64-objs5-minobj10-maxobj15 \
        --bs 64,256,1024 --train-steps 0
--train-steps N first advances the model N recipe steps so the measurement
sits on the plateau proper rather than at init.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_DIR_RE = (r'sqoop-seed(?P<seed>\d+)-train(?P<train_size>\d+)-test(?P<test_size>\d+)'
           r'-rhs(?P<rhs_variety>\d+)-img(?P<img_size>\d+)-objs(?P<num_objects>\d+)'
           r'-minobj(?P<min_obj_size>\d+)-maxobj(?P<max_obj_size>\d+)')


def ensure_dataset(root, d):
    """Generate the dataset in place when the dir's npz files are missing.
    The canonical dir name carries the full config, so nothing else is
    needed; unparseable names fail with a clear message instead."""
    import re
    if os.path.exists(os.path.join(root, d, 'train.npz')):
        return
    m = re.fullmatch(_DIR_RE + r'(?P<suffix>.*)', d)
    if not m:
        raise SystemExit(f'{d}/train.npz is missing and the dir name does not '
                         'match the canonical pattern, so it cannot be generated here. '
                         'Generate it first (tools/prepare_all.py) or pass a canonical dir.')
    from src.tasks import TASKS
    from src.tasks.sqoop.config import SqoopDataConfig
    kw = {k: int(v) for k, v in m.groupdict().items() if k != 'suffix'}
    print(f'== dataset missing: generating {d} '
          f'({kw["train_size"]:,} examples, ~{kw["train_size"] * kw["img_size"] ** 2 * 3 / 1e9:.1f} GB, '
          'roughly an hour single-core at full size) ==')
    TASKS['sqoop'].prepare(SqoopDataConfig(root=root, dir=d, name='sqoop', **kw))
    print('== generation done ==\n')


def load_split(root, d, split):
    z = np.load(os.path.join(root, d, f'{split}.npz'))
    return z['images'], z['questions'], z['answers']


def data_checks(root, d):
    print('== data checks ==')
    tr_q = tr_a = None
    for split in ['train', 'val_seen', 'val_unseen', 'test_unseen']:
        try:
            im, q, a = load_split(root, d, split)
        except FileNotFoundError:
            print(f'  {split:12s} MISSING'); continue
        pix = (im.reshape(len(im), -1) > 16).mean()
        triples = set(map(tuple, q.tolist()))
        print(f'  {split:12s} n={len(a):>9,}  answer-mean={a.mean():.4f}  img dtype={im.dtype} '
              f'max={im.max()}  lit-pixel frac={pix:.4f}  distinct questions={len(triples):,}')
        if split == 'train':
            tr_q, tr_a = q, a
            per = {}
            for t, ans in zip(map(tuple, q.tolist()), a.tolist()):
                per.setdefault(t, [0, 0])[ans] += 1
            worst = max(per.values(), key=lambda v: abs(v[0] - v[1]) / max(sum(v), 1))
            print(f'    per-question balance: worst |pos-neg|/n = {abs(worst[0]-worst[1])/sum(worst):.3f} over {len(per):,} triples')
            lhs_rhs = {}
            for x, r, y in per:
                lhs_rhs.setdefault((x, r), set()).add(y)
            ks = sorted({len(v) for v in lhs_rhs.values()})
            print(f'    rhs per (lhs, rel): {ks[:3]}{"..." if len(ks) > 3 else ""} (should be the rhs_variety, uniform)')
        elif 'unseen' in split and tr_q is not None:
            leak = triples & set(map(tuple, tr_q.tolist()))
            print(f'    LEAK CHECK: {len(leak)} unseen triples occur in train {"  <-- PROBLEM" if leak else "(clean)"}')


def grad_checks(root, d, bss, train_steps, device, reps, seed=0):
    print(f'\n== gradient checks (field-bus SyncNet, sqoop/busnet/stim recipe, eager) | model seed {seed} ==')
    from hydra import compose, initialize_config_dir
    from src import register_configs
    from src.core.registry import build_model
    register_configs()
    with initialize_config_dir(config_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'conf'), version_base='1.3'):
        cfg = compose(config_name='config', overrides=['task=sqoop', 'experiment=sqoop/busnet/stim',
                                                       f'dataset.root={root}', f'dataset.dir={d}', f'train.seed={seed}'])
    torch.manual_seed(seed)
    m = build_model(cfg).to(device)
    im, q, a = load_split(root, d, 'train')
    order = np.random.default_rng(0).permutation(len(a))

    def batch(bs, i=0):
        idx = order[(i * bs) % max(1, len(a) - bs):][:bs]
        return {'images': torch.tensor(im[idx]).float().permute(0, 3, 1, 2).div_(255).to(device),
                'questions': torch.tensor(q[idx]).to(device),
                'answers': torch.tensor(a[idx]).long().to(device)}

    def gradvec(b):
        m.zero_grad(set_to_none=True)
        F.cross_entropy(m(b)['logits'], b['answers']).backward()
        return torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])

    if train_steps:
        opt = torch.optim.AdamW(m.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        for i in range(train_steps):
            b = batch(int(cfg.train.train_bs), i)
            m.zero_grad(set_to_none=True)
            F.cross_entropy(m(b)['logits'], b['answers']).backward()
            opt.step()
        print(f'  advanced {train_steps} recipe steps before measuring')

    print(f'  {"bs":>6s} {"batch |g|":>12s}  (pure cancellation ~ 1/sqrt(B): expect ratio '
          f'{ [round((max(bss)/b)**.5, 1) for b in bss] } vs largest)')
    norms = {}
    for bs in bss:
        ns = [gradvec(batch(bs, r)).norm().item() for r in range(reps)]
        norms[bs] = float(np.mean(ns))
        print(f'  {bs:>6d} {norms[bs]:>12.3e}')
    big = max(bss)
    print('  measured ratios vs largest:', {b: round(norms[b] / norms[big], 2) for b in bss})

    cs = []
    for r in range(reps):
        b = batch(2 * min(bss), 10 + r)
        h = min(bss)
        ga = gradvec({k: v[:h] for k, v in b.items()})
        gb = gradvec({k: v[h:2 * h] for k, v in b.items()})
        cs.append(F.cosine_similarity(ga, gb, dim=0).item())
    print(f'  half-batch grad cosine: {np.mean(cs):+.3f} +- {np.std(cs):.3f}   '
          '(~0 = signal-free plateau; clearly > 0 = signal present)')

    pe = [gradvec(batch(1, 50 + r)).norm().item() for r in range(min(8, reps * 2))]
    print(f'  per-example |g|: {np.mean(pe):.3e}  (healthy if ~100-1000x the large-batch norm on a plateau)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='./data')
    ap.add_argument('--dir', required=True)
    ap.add_argument('--bs', default='64,256,1024')
    ap.add_argument('--train-steps', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0, help='model init/train seed (diagnose the failed seeds at matched steps)')
    ap.add_argument('--reps', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    ensure_dataset(a.root, a.dir)
    data_checks(a.root, a.dir)
    grad_checks(a.root, a.dir, [int(x) for x in a.bs.split(',')], a.train_steps, a.device, a.reps, a.seed)