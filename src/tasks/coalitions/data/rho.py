"""Graph realisability residual `rho(G, d)` for oscillator gates.

An oscillator gate places module k at a unit vector z_k on the sphere S^{d-1}
and opens channel (i, j) with a *sharpened* cosine strength

    g_ij = sigmoid( alpha * <z_i, z_j> + b )   in (0, 1),

matching the model's gate (see gates.py). The sharpening (alpha, b) matters:
the raw cosine gate (1 + cos)/2 cannot fully *close* a channel (orthogonal
placements give 0.5, antipodal is needed for 0), so without sharpening even the
EMPTY and single-PAIR graphs show large residuals that reflect the gate's
inability to close rather than any topological frustration. Optimising over
(alpha, b) removes that confound and isolates the triangle-inequality
constraint we actually care about.

`rho(G, d)` is the smallest RMS error, over all placements {z_k} and sharpening
(alpha, b), between the realised gate matrix and the target adjacency of G:

    rho(G, d) = min_{z_k in S^{d-1}, alpha > 0, b}
                sqrt( mean_{i<j} ( g_ij - 1[(i,j) in G] )^2 ).

Interpretation:
  * rho ~ 0  -> G is (approximately) realisable at dimension d. Every
               clusterable graph -- disjoint unions of cliques, incl. EMPTY and
               MATCH -- is realisable with a scalar phase, d = 2.
  * rho > 0  -> G is frustrated at dimension d; rho lower-bounds the gate error
               a phase model of that dimension must incur. Sharpening cannot
               remove it because the constraint is topological: if A is near B
               and B near C then A is near C, whatever alpha, b are.

This is the x-axis of the calibration plot: on the toy task, phase-model task
error should climb with rho(G, d) while unconstrained-gate error stays flat.
As d grows the constraint relaxes and rho(G, d) -> 0 (the dimension ladder),
with attention as the d -> infinity limit.

The minimisation is non-convex; we use multi-start Adam on unit vectors. The
graphs are tiny (N <= 4) so this is milliseconds and is computed once at
dataset-prep time, then cached in the .npz.
"""

from __future__ import annotations

import torch

from .graphs import Graph, catalogue, n_pairs, pair_index


def _target_pairs(graph: Graph) -> torch.Tensor:
    idx = pair_index(graph.n)
    t = torch.zeros(n_pairs(graph.n))
    for e in graph.edges:
        t[idx[e]] = 1.0
    return t


def rho(
        graph: Graph,
        d: int,
        *,
        n_starts: int = 16,
        n_steps: int = 400,
        lr: float = 0.1,
        seed: int = 0,
        ) -> float:
    """Minimum RMS gate residual for graph at oscillator dimension d."""
    n = graph.n
    if n_pairs(n) == 0:
        return 0.0

    pairs = list(pair_index(n).keys())
    ii = torch.tensor([i for (i, _) in pairs])
    jj = torch.tensor([j for (_, j) in pairs])
    target = _target_pairs(graph)

    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n_starts, n, d, generator=g, requires_grad=True)
    # per-start sharpening: alpha = softplus(raw_alpha) > 0, bias free
    raw_alpha = torch.full((n_starts, 1), 2.0, requires_grad=True)
    bias = torch.zeros(n_starts, 1, requires_grad=True)

    opt = torch.optim.Adam([z, raw_alpha, bias], lr=lr)

    def gate_of(z_):
        zn = torch.nn.functional.normalize(z_, dim=-1)
        dots = (zn[:, ii, :] * zn[:, jj, :]).sum(-1)          # (starts, pairs)
        alpha = torch.nn.functional.softplus(raw_alpha)
        return torch.sigmoid(alpha * dots + bias)

    for _ in range(n_steps):
        opt.zero_grad()
        err = ((gate_of(z) - target) ** 2).mean(dim=-1)       # per start
        err.sum().backward()
        opt.step()

    with torch.no_grad():
        err = ((gate_of(z) - target) ** 2).mean(dim=-1)
        best = err.min().sqrt().item()
    return best


def rho_table(
        n: int,
        d: int,
        **kwargs,
        ) -> dict[str, float]:
    """rho(G, d) for every named graph of a given module count."""
    return {g.name: rho(g, d, **kwargs) for g in catalogue(n)}


if __name__ == '__main__':
    # quick sanity check across the dimension ladder
    for n in (2, 4):
        print(f'--- N = {n} ---')
        for d in (2, 3, 4, 8):
            tbl = rho_table(n, d, n_starts=24, n_steps=500)
            row = '  '.join(f'{k}={v:.3f}' for k, v in tbl.items())
            print(f'd={d}: {row}')
