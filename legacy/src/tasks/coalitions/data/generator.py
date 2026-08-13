"""Generator for the `coalitions` task.

Each example is N parallel token streams following independent local rules,
overlaid with a schedule of communication episodes. During an episode a named
graph G is active; a module whose graph-neighbourhood is non-empty must output
the modular sum of its own token and its neighbours' tokens (so unwanted
message arrival corrupts the target -- this enforces the "off" edges). Between
episodes every module predicts its own next token.

Everything a metric needs is baked into the .npz at prep time:
  streams     (n, T, N)   int64   data tokens
  commands    (n, T)      int64   shared command channel (sparse or dense)
  targets     (n, T, N)   int64   per-module target token
  loss_mask   (n, T, N)   float32 1 where a target is defined (else padded)
  regime      (n, T, N)   int8    REGIME_* code, for metric slicing only
  oracle_adj  (n, T, P)   int8    required adjacency per undirected pair (r_t)
  active_gid  (n, T)      int16   active graph id per step (EMPTY when indep.)
  length      (n,)        int32   true sequence length (<= T if padded)
Plus, once per file, the rho table (used by the calibration callback):
  rho_names   (G,)        str
  rho_d<k>    (G,)        float32  rho(G, k) for k in the configured ladder
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import constants as C
from .graphs import (
    Graph,
    catalogue,
    graph_ids,
    select_family,
    n_pairs,
    pair_index,
    adjacency_vector,
)
from .rho import rho_table

if TYPE_CHECKING:
    from ..config import CoalitionsDataConfig


# --- scheduling --------------------------------------------------------------

def _sample_schedule(
        T: int,
        n_episodes: int,
        win_min: int,
        win_max: int,
        graphs: list[Graph],
        rng: random.Random,
        ) -> list[tuple[int, int, Graph]]:
    """Non-overlapping episodes as (t_on, t_off, graph), left to right."""
    episodes: list[tuple[int, int, Graph]] = []
    cursor = rng.randint(1, max(1, T // 4))     # start independent
    for _ in range(n_episodes):
        win = rng.randint(win_min, win_max)
        if cursor + win + 2 >= T:               # leave a tail step for targets
            break
        graph = rng.choice(graphs)
        episodes.append((cursor, cursor + win, graph))
        cursor = cursor + win + rng.randint(1, max(1, (T - cursor) // 3))
        if cursor >= T - 2:
            break
    return episodes


def _active_graph_at(
        episodes: list[tuple[int, int, Graph]], t: int
        ) -> Graph | None:
    for (t_on, t_off, g) in episodes:
        if t_on <= t < t_off:
            return g
    return None


# --- one example -------------------------------------------------------------

def _make_example(
        cfg_n: int,
        K: int,
        T: int,
        increments: list[int],
        episodes_range: tuple[int, int],
        window_range: tuple[int, int],
        graphs: list[Graph],
        command_mode: str,
        readout_mode: str,
        readout_lag_max: int,
        post_steps: int,
        stream_mode: str,
        gids: dict[str, int],
        pidx: dict[tuple[int, int], int],
        rng: random.Random,
        np_rng: np.random.Generator,
        ) -> dict[str, np.ndarray]:

    N = cfg_n
    P = n_pairs(N)
    pad = C.pad_id(K)

    # local streams
    streams = np.full((T, N), pad, dtype=np.int64)
    if stream_mode == 'rule':
        # deterministic x_{t+1} = (x_t + c_k) mod K. NOTE: this makes a
        # neighbour's current token partially inferable from history, letting a
        # module dodge live routing. Use 'iid' for the routing experiment.
        for k in range(N):
            c = rng.choice(increments)
            x0 = int(np_rng.integers(0, K))
            streams[:, k] = (x0 + c * np.arange(T)) % K
    elif stream_mode == 'iid':
        # high-entropy: each token i.i.d. uniform, so a neighbour's CURRENT
        # token cannot be predicted from history -- it must be routed live.
        streams[:] = np_rng.integers(0, K, size=(T, N))
    else:
        raise ValueError(f'unknown stream_mode {stream_mode!r}')

    n_ep = rng.randint(episodes_range[0], episodes_range[1])
    episodes = _sample_schedule(
        T, n_ep, window_range[0], window_range[1], graphs, rng
    )

    commands = np.full(T, C.NOOP, dtype=np.int64)
    active_gid = np.full(T, gids['EMPTY'], dtype=np.int16)
    oracle_adj = np.zeros((T, P), dtype=np.int8)

    for (t_on, t_off, g) in episodes:
        gid = gids[g.name]
        adj = np.array(adjacency_vector(g), dtype=np.int8)
        for t in range(t_on, t_off):
            active_gid[t] = gid
            oracle_adj[t] = adj
        if command_mode == 'sparse':
            commands[t_on] = C.connect_id(gid)
            if t_off < T:
                commands[t_off] = C.DISCONNECT
        elif command_mode == 'dense':
            for t in range(t_on, t_off):
                commands[t] = C.hold_id(gid)
            if t_off < T:
                commands[t_off] = C.DISCONNECT
        else:
            raise ValueError(f'unknown command_mode {command_mode!r}')

    targets = np.full((T, N), 0, dtype=np.int64)   # masked positions -> 0
    loss_mask = np.zeros((T, N), dtype=np.float32)
    regime = np.full((T, N), C.REGIME_INDEP, dtype=np.int8)

    # mark POST steps (just after each disconnect)
    post_flag = np.zeros(T, dtype=bool)
    for (_, t_off, _) in episodes:
        post_flag[t_off: min(T, t_off + post_steps)] = True

    def combine(t: int, k: int, g: Graph) -> int:
        # PLAIN integer sum (no modulo): a routing-limited readout. Any unwanted
        # arrival changes the sum, so off-edges are still enforced, but the
        # target is ordinary small-integer addition -- learnable, so routing
        # (not the readout) is the bottleneck and the oracle can approach 100%.
        nbrs = g.neighbours(k)
        s = int(streams[t, k])
        for j in nbrs:
            s += int(streams[t, j])
        return s

    if readout_mode == 'instant':
        for t in range(T):
            g = _active_graph_at(episodes, t)
            for k in range(N):
                nbrs = () if g is None else g.neighbours(k)
                if nbrs:
                    targets[t, k] = combine(t, k, g)   # type: ignore[arg-type]
                    loss_mask[t, k] = 1.0
                    regime[t, k] = C.REGIME_JOINT
                else:
                    # independent regime. 'iid' -> copy own CURRENT token (a
                    # trivial no-routing default); 'rule' -> predict next token.
                    if stream_mode == 'iid':
                        targets[t, k] = streams[t, k]
                        loss_mask[t, k] = 1.0
                    elif t + 1 < T:
                        targets[t, k] = streams[t + 1, k]
                        loss_mask[t, k] = 1.0
                    regime[t, k] = (
                        C.REGIME_POST if post_flag[t] else C.REGIME_INDEP
                    )

    elif readout_mode == 'latch':
        # payload latched at t_on; combine emitted at a cue step after t_off.
        for t in range(T):
            for k in range(N):
                if t + 1 < T:
                    targets[t, k] = streams[t + 1, k]
                    loss_mask[t, k] = 1.0
                regime[t, k] = (
                    C.REGIME_POST if post_flag[t] else C.REGIME_INDEP
                )
        for (t_on, t_off, g) in episodes:
            lag = rng.randint(0, readout_lag_max)
            t_cue = min(T - 1, t_off + lag)
            for k in range(N):
                nbrs = g.neighbours(k)
                if not nbrs:
                    continue
                s = int(streams[t_on, k])
                for j in nbrs:
                    s += int(streams[t_on, j])
                targets[t_cue, k] = s               # plain integer sum
                loss_mask[t_cue, k] = 1.0
                regime[t_cue, k] = C.REGIME_READOUT
    else:
        raise ValueError(f'unknown readout_mode {readout_mode!r}')

    return {
        'streams': streams,
        'commands': commands,
        'targets': targets,
        'loss_mask': loss_mask,
        'regime': regime,
        'oracle_adj': oracle_adj,
        'active_gid': active_gid,
        'length': np.int32(T),
    }


# --- build & save ------------------------------------------------------------

def build_split(
        size: int, cfg: CoalitionsDataConfig, T: int, seed: int,
        ) -> dict[str, np.ndarray]:

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    gids = graph_ids(cfg.n_modules)
    pidx = pair_index(cfg.n_modules)
    graphs = select_family(cfg.n_modules, cfg.family)
    increments = list(cfg.increments)

    keys = [
        'streams', 'commands', 'targets', 'loss_mask',
        'regime', 'oracle_adj', 'active_gid', 'length',
    ]
    acc: dict[str, list] = {k: [] for k in keys}

    for _ in range(size):
        ex = _make_example(
            cfg_n=cfg.n_modules,
            K=cfg.K,
            T=T,
            increments=increments,
            episodes_range=(cfg.episodes_min, cfg.episodes_max),
            window_range=(cfg.window_min, cfg.window_max),
            graphs=graphs,
            command_mode=cfg.command_mode,
            readout_mode=cfg.readout_mode,
            readout_lag_max=cfg.readout_lag_max,
            post_steps=cfg.post_steps,
            stream_mode=cfg.stream_mode,
            gids=gids,
            pidx=pidx,
            rng=rng,
            np_rng=np_rng,
        )
        for k in keys:
            acc[k].append(ex[k])

    out = {k: np.stack(acc[k]) for k in keys if k != 'length'}
    out['length'] = np.array(acc['length'], dtype=np.int32)
    return out


def _attach_rho(data: dict[str, np.ndarray], cfg: CoalitionsDataConfig) -> None:
    cat = catalogue(cfg.n_modules)
    data['rho_names'] = np.array([g.name for g in cat])
    for d in cfg.rho_dims:
        tbl = rho_table(cfg.n_modules, int(d))
        data[f'rho_d{int(d)}'] = np.array(
            [tbl[g.name] for g in cat], dtype=np.float32
        )


def _save(data: dict[str, np.ndarray], data_dir: Path, name: str) -> None:
    data_dir.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(data_dir / name, **data)
    print(f'saved {name} to {data_dir.absolute()}')


def prepare_coalitions(cfg: CoalitionsDataConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    data_dir = Path(cfg.root) / cfg.dir

    print('building train split...')
    train = build_split(cfg.train_size, cfg, cfg.T_train, cfg.seed + 1)
    _attach_rho(train, cfg)
    _save(train, data_dir, 'train.npz')

    print('building val split...')
    val = build_split(cfg.test_size, cfg, cfg.T_train, cfg.seed + 2)
    _attach_rho(val, cfg)
    _save(val, data_dir, 'val.npz')

    print('building test split...')
    # test uses the generalisation length T_test (defaults to T_train)
    test = build_split(cfg.test_size, cfg, cfg.T_test, cfg.seed + 3)
    _attach_rho(test, cfg)
    _save(test, data_dir, 'test.npz')
