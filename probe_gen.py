"""Generate a small but faithful SQOOP sample using the repo's own generator.

Bypasses hydra: builds the schedule + splits directly so we can control
rhs_variety and size without a 40-minute build.
"""
import sys, time, argparse
import numpy as np
sys.path.insert(0, '/root/proj')

from src.tasks.sqoop.data.generator import (
    _build_schedule, _gen_split, _repeats_for,
)
from src.tasks.sqoop.data.constants import SHAPES
import random

p = argparse.ArgumentParser()
p.add_argument('--rhs', type=int, default=18)
p.add_argument('--n', type=int, default=40000)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', type=str, required=True)
a = p.parse_args()

base_seed = a.seed
py_rng = random.Random(base_seed)
train_pairs = []
for i, x in enumerate(SHAPES):
    for y in py_rng.sample(SHAPES[:i] + SHAPES[i + 1:], a.rhs):
        train_pairs.append((x, y))

reps = _repeats_for(a.n, len(train_pairs), 'train_size')
sched_rng = np.random.RandomState(base_seed)
sched = _build_schedule(train_pairs, reps, sched_rng)
print(f'rhs={a.rhs}: {len(train_pairs)} pairs, {reps} reps/pair, '
      f'{len(sched):,} examples')

t0 = time.time()
arrays = _gen_split(sched, base_seed + 1, num_objects=5, img_size=64,
                    min_obj=10, max_obj=15, restrict_positive=False)
print(f'generated in {time.time()-t0:.1f}s')
np.savez_compressed(a.out, **arrays)
print('saved', a.out)
