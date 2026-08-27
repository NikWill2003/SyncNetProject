#!/usr/bin/env bash
set -uo pipefail

# PIXEL BUSNET, ROUND 2, GPU 1 -- the all-synchrony front end. 12 runs,
# ~15 h (field runs take ~85 min). field_stim6 seed 0 gave .923 / .784
# ternary, the best pixel number in the project; before that is believed
# it needs seeds and the three controls it never had.
#
# READ: field_stim6 (3 seeds) against field_static6 (same capacity, stored
# addresses), field_full (private-line ceiling with the same perception),
# field_zero (no communication: what the field computes on its own) and
# field_open (seed 0 was .682 -- seed lottery or real?). Then d=8.

BUDGET_S=$((24 * 3600)); EST_S=$((85 * 60)); mkdir -p logs/sync_d
run() { local exp="$1" stem="$2"; shift 2
  if (( SECONDS + EST_S > BUDGET_S )); then echo "[skip] ${exp} ${*:-}"; return 0; fi
  python main.py task=sort_of_clevr experiment=sort_of_clevr/sync_d/${exp} "$@" 2>&1 | tee logs/sync_d/${stem}.log; }

run field_stim6   10_field_stim6_s1   train.seed=1
run field_static6 11_field_static6_s0 train.seed=0
run field_zero    12_field_zero_s0    train.seed=0
run field_full    13_field_full_s0    train.seed=0
run field_stim6   14_field_stim6_s2   train.seed=2
run field_open    15_field_open_s1    train.seed=1
run field_static6 16_field_static6_s1 train.seed=1
run field_stim8   17_field_stim8_s0   train.seed=0
run field_full    18_field_full_s1    train.seed=1
run field_static6 19_field_static6_s2 train.seed=2
run field_zero    20_field_zero_s1    train.seed=1
run field_stim8   21_field_stim8_s1   train.seed=1

echo; echo "elapsed: $(( SECONDS / 60 )) min"; echo "failures:"; grep -lEi "Traceback|Error executing job" logs/sync_d/*.log || echo "  none"
