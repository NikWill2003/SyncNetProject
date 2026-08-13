"""Constants for the `coalitions` task.

Mechanism-neutral synthetic sequence task for studying whether a learned
communication gate (phase / synchrony, or a conventional learned gate) can
open and close channels between recurrent modules on demand.

Vocabulary layout
------------------
Data tokens occupy ids [0, K). Special tokens are appended after the K data
tokens so a single embedding table covers everything:

    PAD   : padding for variable-length sequences (masked out of the loss)

Command tokens live in a *separate* small vocabulary (see COMMANDS) and are
embedded by their own table, because the command channel is conceptually
distinct from the data channel and we sometimes want it sparse (mostly NOOP).

Regime codes (per timestep, per module) are used only for metric slicing;
they are never seen by the model.
"""

from __future__ import annotations

# ---- data-token vocabulary --------------------------------------------------

# K (alphabet size) is a dataset-config field; PAD is the single extra id.
# vocab_size passed to the model = K + N_EXTRA_TOKENS.
N_EXTRA_TOKENS = 1
PAD_OFFSET = 0  # PAD id = K + PAD_OFFSET


def pad_id(K: int) -> int:
    return K + PAD_OFFSET


def vocab_size(K: int) -> int:
    return K + N_EXTRA_TOKENS


def readout_vocab_size(n_modules: int, K: int) -> int:
    """Output-head cardinality for the integer-sum readout.

    Joint targets are the plain (non-modular) sum of a module's own token and
    its neighbours' tokens; the largest possible sum is over a fully-connected
    module, n_modules * (K - 1). Independent-step targets are in [0, K), which
    this range also covers. So one head of size n_modules * (K - 1) + 1 serves
    both regimes exactly.
    """
    return n_modules * (K - 1) + 1


# ---- command vocabulary -----------------------------------------------------
# Sparse mode emits CONNECT_<gid>/DISCONNECT only at episode edges and NOOP
# elsewhere. Dense mode emits HOLD_<gid> every step (the active graph id).
#
# We keep a fixed command table sized for the largest graph catalogue we use.
# Command ids:
#   0            : NOOP            (no edge event this step)
#   1            : DISCONNECT      (an episode ended this step)
#   2 + gid      : CONNECT graph gid   (sparse: emitted at t_on)
#   2 + G + gid  : HOLD    graph gid   (dense:  emitted while gid is active)
#
# G = MAX_GRAPHS is an upper bound on the number of named graphs across N.

NOOP = 0
DISCONNECT = 1
CONNECT_BASE = 2
MAX_GRAPHS = 16
HOLD_BASE = CONNECT_BASE + MAX_GRAPHS

N_COMMANDS = HOLD_BASE + MAX_GRAPHS


def connect_id(gid: int) -> int:
    return CONNECT_BASE + gid


def hold_id(gid: int) -> int:
    return HOLD_BASE + gid


# ---- regime codes (metric slicing only) ------------------------------------

REGIME_INDEP = 0      # module has no active neighbours -> local next-token
REGIME_JOINT = 1      # module has active neighbours -> combine target
REGIME_READOUT = 2    # latch mode: the cue step where the combine is emitted
REGIME_POST = 3       # first few steps immediately after a DISCONNECT
REGIME_PAD = 4        # padded step (masked)

N_REGIMES = 5
