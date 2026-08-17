"""Is the SQOOP label predictable from the IMAGE ALONE (plus the relation
token, which carries no binding information)?

The generator's own docstring flags an unfixed asymmetry: negatives are
rejection-sampled under three simultaneous positional constraints
(NOT x rel y, x rel obj4, obj3 rel y) while positives are constrained
only by x rel y. That biases the LAYOUT of negatives in a way no
positive shares -- and layout is exactly what a spatially-unrestricted
CNN readout reads.

If an image-only predictor beats chance by a lot, then a 0.999 conv_lstm
on test_unseen is not evidence of systematic relational generalisation,
and the whole SQOOP baseline needs a caveat.
"""
import sys
import numpy as np

sys.path.insert(0, '/root/proj')
from src.tasks.sqoop.data.constants import N_SHAPES, RELATIONS

path = sys.argv[1] if len(sys.argv) > 1 else '/root/proj/probe/rhs18.npz'
NMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 24000

d = np.load(path)
imgs = d['images'][:NMAX]
qs = d['questions'][:NMAX]
ans = d['answers'][:NMAX].astype(np.float64)
N = imgs.shape[0]
rel = qs[:, 1] - N_SHAPES
print(f'{path}: N={N}, positive rate {ans.mean():.4f}')

ink = (imgs.sum(-1) > 0).astype(np.float32)          # (N, 64, 64)
col = ink.sum(1)                                      # (N, 64) ink per x
row = ink.sum(2)                                      # (N, 64) ink per y
grid = ink.reshape(N, 8, 8, 8, 8).mean(axis=(2, 4))   # (N, 8, 8) coarse map

xs = np.arange(64, dtype=np.float32)


def moments(prof):
    w = prof / np.maximum(prof.sum(1, keepdims=True), 1e-6)
    m1 = (w * xs).sum(1)
    m2 = np.sqrt(np.maximum((w * (xs - m1[:, None]) ** 2).sum(1), 0))
    m3 = (w * (xs - m1[:, None]) ** 3).sum(1) / np.maximum(m2 ** 3, 1e-6)
    return np.stack([m1, m2, m3, prof.sum(1)], 1)


feat_global = np.concatenate([moments(col), moments(row)], 1)   # (N, 8)
feat_grid = grid.reshape(N, 64)


def logreg(X, y, iters=400, lr=0.5, l2=1e-4, split=0.7):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    X = np.concatenate([X, np.ones((len(X), 1))], 1)
    n_tr = int(len(X) * split)
    Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xtr @ w))
        w -= lr * (Xtr.T @ (p - ytr) / len(Xtr) + l2 * w)
    acc_tr = (((Xtr @ w) > 0) == (ytr > 0.5)).mean()
    acc_te = (((Xte @ w) > 0) == (yte > 0.5)).mean()
    return acc_tr, acc_te


print('\n--- image-only predictors (no shape identity, no binding) ---')
for label, F in [('8 global layout moments', feat_global),
                 ('8x8 coarse ink map', feat_grid),
                 ('both', np.concatenate([feat_global, feat_grid], 1))]:
    accs = []
    for r, rname in enumerate(RELATIONS):
        m = rel == r
        a_tr, a_te = logreg(F[m], ans[m])
        accs.append(a_te)
        print(f'  {label:24s} rel={rname:9s} n={m.sum():6d} '
              f'train {a_tr:.4f}  held-out {a_te:.4f}')
    print(f'  {label:24s} MEAN held-out over relations: {np.mean(accs):.4f}\n')

# how much does the number of ink pixels alone say?
for r, rname in enumerate(RELATIONS):
    m = rel == r
    t = ink[m].sum(axis=(1, 2))
    print(f'  ink mass  rel={rname:9s} pos {t[ans[m] > .5].mean():8.2f} '
          f'neg {t[ans[m] < .5].mean():8.2f}  '
          f'd={(t[ans[m]>.5].mean()-t[ans[m]<.5].mean())/t.std():+.3f} sd')
