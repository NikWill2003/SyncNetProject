"""Named communication graphs and graph-family catalogues.

A communication graph over N modules is an undirected graph whose edges are the
channels that must be *open* during an episode. The task's per-module targets
sum a module's own token with those of its graph-neighbours, so realising the
wrong graph (e.g. a clique where a path was required) delivers unwanted tokens
and corrupts the receiver -- this is what enforces the "required-off" edges.

The central structural fact under test (corrected against rho.py -- see the
note on families below):

    A sharpened cosine phase gate g_ij = sigmoid(alpha cos(phi_i - phi_j) + b)
    places modules on a circle and opens a channel when two points fall within
    an angular threshold. Such a gate realises *circular-arc threshold* graphs,
    which is a broader class than disjoint unions of cliques: a PATH embeds
    fine as an arc (A,B,C,D at 0,60,120,180 deg puts edges at cos 60 = 0.5 and
    non-edges at cos >= 120 deg), and a 4-CYCLE embeds as a square. These are
    NOT frustrated at d = 2.

    The genuinely frustrated structure at d = 2 is the STAR K_{1,m} (m >= 3): a
    hub adjacent to several mutually non-adjacent leaves. Placing the hub near
    all leaves forces the leaves into a small arc, so they cannot stay
    non-adjacent. rho(STAR_A, 2) ~ 0.41, collapsing to ~0.02 at d = 3 (three
    dimensions give room to spread the spokes). Larger stars K_{1,m} need
    proportionally higher dimension, which is why a graded ladder needs N >= 6.

Family tags below are assigned from the measured rho(G, 2) in rho.py, NOT from
a clique-union heuristic:
  * 'clusterable'  : rho(G, 2) ~ 0        (realisable by scalar phase)
  * 'nonclique'    : rho(G, 2) ~ 0 but G is not a union of cliques
                     (realisable arc/cycle; useful intermediate controls)
  * 'frustrated'   : rho(G, 2) notably > 0 (star-like; the discriminating case)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class Graph:
    name: str
    n: int
    edges: tuple[tuple[int, int], ...]   # undirected, i < j
    family: str                          # 'clusterable' | 'frustrated' | 'empty'

    def neighbours(self, k: int) -> tuple[int, ...]:
        out = []
        for (i, j) in self.edges:
            if i == k:
                out.append(j)
            elif j == k:
                out.append(i)
        return tuple(sorted(out))

    def edge_set(self) -> frozenset[tuple[int, int]]:
        return frozenset(self.edges)


def _all_pairs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i, j in combinations(range(n), 2)]


# --- N = 2 -------------------------------------------------------------------
# Only INDEPENDENT (empty) and COMBINE (the single pair) exist.

_N2 = [
    Graph('EMPTY', 2, (), 'empty'),
    Graph('COMBINE', 2, ((0, 1),), 'clusterable'),
]

# --- N = 4 -------------------------------------------------------------------
# Modules labelled A=0, B=1, C=2, D=3.

_N4 = [
    Graph('EMPTY', 4, (), 'empty'),
    # clusterable: disjoint unions of cliques, rho(.,2) ~ 0
    Graph('PAIR_AB', 4, ((0, 1),), 'clusterable'),
    Graph('PAIR_CD', 4, ((2, 3),), 'clusterable'),
    Graph('MATCH', 4, ((0, 1), (2, 3)), 'clusterable'),
    Graph('TRIANGLE_ABC', 4, ((0, 1), (1, 2), (0, 2)), 'clusterable'),
    Graph('FULL', 4, tuple(_all_pairs(4)), 'clusterable'),
    # nonclique but realisable at d = 2 (arc / cycle), rho(.,2) ~ 0
    Graph('PATH_ABCD', 4, ((0, 1), (1, 2), (2, 3)), 'nonclique'),
    Graph('CYCLE4', 4, ((0, 1), (1, 2), (2, 3), (0, 3)), 'nonclique'),
    # frustrated at d = 2, realisable at d >= 3
    Graph('STAR_A', 4, ((0, 1), (0, 2), (0, 3)), 'frustrated'),
]

# --- N = 6 -------------------------------------------------------------------
# Gives a graded frustration ladder: bigger stars need higher dimension, and
# complete-bipartite cores K_{2,3}/K_{3,3} are frustrated at d = 2 too. Modules
# A..F = 0..5.

def _star(n: int, hub: int) -> tuple[tuple[int, int], ...]:
    return tuple((min(hub, k), max(hub, k)) for k in range(n) if k != hub)


def _complete_bipartite(left: tuple[int, ...], right: tuple[int, ...]):
    return tuple((min(i, j), max(i, j)) for i in left for j in right)


_N6 = [
    Graph('EMPTY', 6, (), 'empty'),
    # clusterable
    Graph('THREE_PAIRS', 6, ((0, 1), (2, 3), (4, 5)), 'clusterable'),
    Graph('TWO_TRIANGLES', 6,
          ((0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)), 'clusterable'),
    Graph('FULL', 6, tuple(_all_pairs(6)), 'clusterable'),
    # nonclique-realisable controls
    Graph('PATH6', 6, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)), 'nonclique'),
    Graph('CYCLE6', 6,
          ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)), 'nonclique'),
    # graded frustration
    Graph('STAR3', 6, _star(6, 0)[:3], 'frustrated'),        # K_{1,3}
    Graph('STAR4', 6, _star(6, 0)[:4], 'frustrated'),        # K_{1,4}
    Graph('STAR5', 6, _star(6, 0), 'frustrated'),            # K_{1,5}
    Graph('BIPARTITE_2_3', 6,
          _complete_bipartite((0, 1), (2, 3, 4)), 'frustrated'),   # K_{2,3}
    Graph('BIPARTITE_3_3', 6,
          _complete_bipartite((0, 1, 2), (3, 4, 5)), 'frustrated'),  # K_{3,3}
]

_CATALOGUE: dict[int, list[Graph]] = {2: _N2, 4: _N4, 6: _N6}


def catalogue(n: int) -> list[Graph]:
    if n not in _CATALOGUE:
        raise ValueError(
            f'no graph catalogue for n_modules={n} '
            f'(supported: {sorted(_CATALOGUE)})'
        )
    return _CATALOGUE[n]


def graph_by_name(n: int, name: str) -> Graph:
    for g in catalogue(n):
        if g.name == name:
            return g
    raise ValueError(f'no graph named {name!r} for n_modules={n}')


def graph_ids(n: int) -> dict[str, int]:
    """Stable gid per graph name (index into the catalogue)."""
    return {g.name: i for i, g in enumerate(catalogue(n))}


def select_family(n: int, family: str) -> list[Graph]:
    """Non-empty graphs to sample episodes from.

    family: 'clusterable' | 'nonclique' | 'frustrated' | 'all'. The EMPTY graph
    is the between-episode default and is never sampled as an episode.
    """
    cat = [g for g in catalogue(n) if g.family != 'empty']
    if family == 'all':
        return cat
    picked = [g for g in cat if g.family == family]
    if not picked:
        raise ValueError(
            f'no graphs in family {family!r} for n_modules={n}'
        )
    return picked


def n_pairs(n: int) -> int:
    return n * (n - 1) // 2


def pair_index(n: int) -> dict[tuple[int, int], int]:
    """Map an undirected pair (i<j) to a flat index in [0, n_pairs)."""
    return {pair: idx for idx, pair in enumerate(_all_pairs(n))}


def adjacency_vector(graph: Graph) -> list[int]:
    """Flat 0/1 vector over undirected pairs for this graph."""
    idx = pair_index(graph.n)
    vec = [0] * n_pairs(graph.n)
    for e in graph.edges:
        vec[idx[e]] = 1
    return vec
