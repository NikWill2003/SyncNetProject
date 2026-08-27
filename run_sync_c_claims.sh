#!/usr/bin/env bash
set -uo pipefail

# BUSNET CLAIMS PASS -- 12 runs, ~9 h at the measured rates (vector d=6
# ~48 min, circle ~25 min, channels ~25 min). Turns the single-seed
# positive (d=6 + stimulus: .916 / .769 ternary vs .851 / .616 for fixed
# axes of the same capacity) into three-seed cells, adds the control that
# was missing (full-access private lines: the real bandwidth ceiling), and
# three follow-ups. Same recipe as sync_c (100k steps, full analysis bundle).
#
# READ: ternary accuracy with mean +- s.d. over seeds for stim6 vs static6
# vs open vs full; eval_model/phase_R and gate_offdiag (the stim6 cells
# should stay unsynchronised, R ~ .5); test_interventions/phase_freeze_drop
# (large for stim6) and phase_shuffle_drop (large for static6, ~0 for
# stim6: computed vs stored addresses).

BUDGET_S=$((12 * 3600))
EST_S=$((50 * 60))
mkdir -p logs/sync_c

run() {  # run <experiment> <log-stem> [extra overrides...]
  local exp="$1" stem="$2"; shift 2
  local n=1; case "$exp" in claims_full) n=3;; claims_*) n=2;; esac
  if (( SECONDS + n * EST_S > BUDGET_S )); then
    echo "[skip] ${exp}: $(( (BUDGET_S - SECONDS) / 60 )) min left, need ~$(( n * EST_S / 60 ))"
    return 0
  fi
  python main.py task=sort_of_clevr experiment=sort_of_clevr/sync_c/${exp} "$@" \
    2>&1 | tee logs/sync_c/${stem}.log
}

#  9 runs ~7 h -- the three-seed cells
run claims_stim6   20_claims_stim6
run claims_static6 21_claims_static6
run claims_full    22_claims_full
run claims_open    23_claims_open

#  3 runs ~2 h -- follow-ups
run stim8              24_stim8
run stim6_msg1         25_stim6_msg1
run circle_phase_rerun 26_circle_phase_rerun

echo; echo "elapsed: $(( SECONDS / 60 )) min"; echo "failures:"
grep -lEi "Traceback|Error executing job" logs/sync_c/*.log || echo "  none"
