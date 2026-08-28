#!/usr/bin/env python3
"""Generate every dataset the final runs need, in parallel, one process per
dataset. Parallelism is ACROSS datasets only: each dataset's generator runs
exactly as it would alone, so every dir is byte-identical to sequential
generation and dirs that already exist (the SoC screen data, the rhs=18
SQOOP set) are detected and skipped, never touched.

    python tools/prepare_all.py --dry-run
    python tools/prepare_all.py --workers 5
    python tools/prepare_all.py --task sqoop --rhs 1,2,4,8 --workers 4
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SOC_SPLITS = ('train.npz', 'val.npz', 'test.npz')
SQOOP_SPLITS = ('train.npz', 'val_seen.npz', 'val_unseen.npz', 'test_unseen.npz')


def soc_cfg(root: str):
    from src.tasks.sort_of_clevr.config import SortOfClevrDataConfig
    c = dict(name='sort_of_clevr', seed=1, train_size=36_000, test_size=1000,
             img_size=75, obj_size=5, nb_questions=10, t_subtype=-1)
    d = ('{name}-seed{seed}-train{train_size}-test{test_size}-img{img_size}'
         '-obj{obj_size}-q{nb_questions}-t{t_subtype}').format(**c)
    return SortOfClevrDataConfig(root=root, dir=d, **c), SOC_SPLITS


def sqoop_cfg(root: str, rhs: int):
    from src.tasks.sqoop.config import SqoopDataConfig
    c = dict(name='sqoop', seed=0, train_size=1_080_000, test_size=25_600,
             rhs_variety=rhs, img_size=64, num_objects=5, min_obj_size=10, max_obj_size=15)
    d = ('{name}-seed{seed}-train{train_size}-test{test_size}-rhs{rhs_variety}'
         '-img{img_size}-objs{num_objects}-minobj{min_obj_size}-maxobj{max_obj_size}').format(**c)
    return SqoopDataConfig(root=root, dir=d, **c), SQOOP_SPLITS


def _generate(task: str, cfg, q) -> None:
    t0 = time.time()
    try:
        from src.tasks import TASKS
        TASKS[task].prepare(cfg)
        q.put((cfg.dir, f'done  ({(time.time() - t0) / 60:.1f} min)'))
    except Exception as e:                                    # noqa: BLE001
        q.put((cfg.dir, f'FAILED  {type(e).__name__}: {e}'))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['soc', 'sqoop', 'all'], default='all')
    ap.add_argument('--rhs', default='1,2,4,8,18')
    ap.add_argument('--root', default='./data')
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    jobs = []
    if a.task in ('soc', 'all'):
        jobs.append(('sort_of_clevr',) + soc_cfg(a.root))
    if a.task in ('sqoop', 'all'):
        for rhs in [int(x) for x in a.rhs.split(',')]:
            jobs.append(('sqoop',) + sqoop_cfg(a.root, rhs))

    todo = []
    for task, cfg, splits in jobs:
        missing = [s for s in splits if not (Path(cfg.root) / cfg.dir / s).exists()]
        state = 'exists, skipped' if not missing else f'MISSING {missing}'
        print(f'  [{state:>28}]  {cfg.dir}')
        if missing:
            todo.append((task, cfg))
    if a.dry_run or not todo:
        print('nothing to do' if not todo else f'{len(todo)} to generate (dry run)')
        return

    workers = min(a.workers, len(todo))
    print(f'\ngenerating {len(todo)} dataset(s) with {workers} processes...')
    ctx = get_context('fork')   # fork: children inherit the target, nothing is pickled
    queue = ctx.Queue()
    pending = list(todo)
    active: list = []
    remaining = len(todo)
    while remaining:
        while pending and len(active) < workers:
            task, cfg = pending.pop(0)
            proc = ctx.Process(target=_generate, args=(task, cfg, queue), daemon=False)
            proc.start()
            active.append(proc)
        name, msg = queue.get()
        print(f'  {msg:<12} {name}', flush=True)
        remaining -= 1
        active = [p for p in active if p.is_alive()]
    for p in active:
        p.join()
    print('all done')


if __name__ == '__main__':
    main()
