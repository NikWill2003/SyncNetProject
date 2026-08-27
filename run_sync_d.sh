#!/usr/bin/env bash
set -uo pipefail

# SYNC SCREEN D -- BusNet on RAW PIXELS. 18 runs, ~18 h at ~60 min each
# (estimate: CNN read on top of the vector d=6 bus). Runs on the other GPU
# while run_sync_c_claims.sh does the object-token claims pass; no cell
# here repeats one there (every sync_c cell is an object-token model).
#
# The model the thesis goes with: content decides what a module holds
# (attention over a CNN grid from the module's own state; colour-keyed
# identities or exchangeable slots), phase decides how modules talk (the
# bus at d=6 with the stimulus drive). Each front end gets the same four
# cells as the object study: stimulus-phase, static addresses (same
# capacity, no dynamics), open bus, full-access private lines. Seed 0 of
# every cell first, then seed 1. Two all-synchrony stretch cells at the end.
#
# READ: ternary accuracy stim6 vs static6 vs open vs full within each front
# end (the object study gave .77 / .62 / .62 / -); test_binding/obj_coverage
# and module_purity (did the modules find the objects: the number that
# bounds everything on pixels); eval_model/phase_R (stim6 should stay
# unsynchronised); test_interventions/phase_freeze_drop and
# phase_shuffle_drop (large / ~0 for stim6, ~0 / large for static6).

BUDGET_S=$((24 * 3600))
EST_S=$((60 * 60))
mkdir -p logs/sync_d

run() {  # run <experiment> <log-stem> [extra overrides...]
  local exp="$1" stem="$2"; shift 2
  if (( SECONDS + EST_S > BUDGET_S )); then
    echo "[skip] ${exp} ${*:-}: $(( (BUDGET_S - SECONDS) / 60 )) min left"
    return 0
  fi
  python main.py task=sort_of_clevr experiment=sort_of_clevr/sync_d/${exp} "$@" \
    2>&1 | tee logs/sync_d/${stem}.log
}

#  8 runs -- seed 0, attention-from-state read then exchangeable slots
for cell in cnn_stim6 cnn_static6 cnn_open cnn_full slots_stim6 slots_static6 slots_open slots_full; do
  run ${cell} 01_${cell}_s0 train.seed=0
done

#  2 runs -- all-synchrony perception (stretch)
run field_stim6 02_field_stim6_s0 train.seed=0
run field_open  02_field_open_s0  train.seed=0

#  8 runs -- seed 1
for cell in cnn_stim6 cnn_static6 cnn_open cnn_full slots_stim6 slots_static6 slots_open slots_full; do
  run ${cell} 03_${cell}_s1 train.seed=1
done

echo; echo "elapsed: $(( SECONDS / 60 )) min"; echo "failures:"
grep -lEi "Traceback|Error executing job" logs/sync_d/*.log || echo "  none"
