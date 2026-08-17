"""Mini-SQOOP: identical semantics (hard negatives, exact per-cell label
balance, position-only labels) but 8 glyphs instead of 36 and 4 objects
instead of 5. Glyph recognition is easy; the relational/binding structure
and the exactly-flat chance plateau are unchanged."""
import sys, time, random, argparse
import numpy as np
sys.path.insert(0, '/root/proj')
import src.tasks.sqoop.data.generator as G
from src.tasks.sqoop.data.generator import (
    _build_schedule, _gen_split, _repeats_for,
)

p = argparse.ArgumentParser()
p.add_argument('--rhs', type=int, default=3)
p.add_argument('--n', type=int, default=40000)
p.add_argument('--shapes', type=int, default=8)
p.add_argument('--objs', type=int, default=4)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', required=True)
a = p.parse_args()

G.SHAPES = list('ABCDEFGHIJKLMNOP')[:a.shapes]
S = G.SHAPES
py_rng = random.Random(a.seed)
pairs = [(x, y) for i, x in enumerate(S)
         for y in py_rng.sample(S[:i] + S[i + 1:], a.rhs)]
reps = _repeats_for(a.n, len(pairs), 'train_size')
sched = _build_schedule(pairs, reps, np.random.RandomState(a.seed))
print(f'mini rhs={a.rhs}: {len(S)} shapes, {len(pairs)} pairs, '
      f'{reps} reps/pair, {len(sched):,} ex')
t0 = time.time()
arr = _gen_split(sched, a.seed + 1, num_objects=a.objs, img_size=64,
                 min_obj=10, max_obj=15, restrict_positive=False)
print(f'generated in {time.time() - t0:.1f}s')
np.savez_compressed(a.out, **arr)
print('saved', a.out)
