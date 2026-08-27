#!/usr/bin/env bash
set -uo pipefail

# PIXEL BUSNET, ROUND 2, GPU 2 -- the attention-from-state front end with the
# read upgraded (four queries per module, competition over modules, re-read
# from the evolving state every step: the v2 SyncNet's read, which is what
# lifted its object coverage). 12 runs, ~12 h. Round 1 with the one-shot
# single-query read had object coverage .60 / purity .40 and the phase
# bus tied the open bus (.690 vs .693 ternary); the question is whether
# better perception restores the ordering seen on object tokens.
#
# READ: test_binding/obj_coverage and module_purity first (did the read
# improve?), then cnn2_stim6 vs cnn2_static6 vs cnn2_open vs cnn2_full at
# two seeds, then d=8 for both reads.

BUDGET_S=$((24 * 3600)); EST_S=$((60 * 60)); mkdir -p logs/sync_d
run() { local exp="$1" stem="$2"; shift 2
  if (( SECONDS + EST_S > BUDGET_S )); then echo "[skip] ${exp} ${*:-}"; return 0; fi
  python main.py task=sort_of_clevr experiment=sort_of_clevr/sync_d/${exp} "$@" 2>&1 | tee logs/sync_d/${stem}.log; }

for cell in cnn2_stim6 cnn2_open cnn2_static6 cnn2_full; do run ${cell} 30_${cell}_s0 train.seed=0; done
run cnn2_stim8 31_cnn2_stim8_s0 train.seed=0
run cnn_stim8  32_cnn_stim8_s0  train.seed=0
for cell in cnn2_stim6 cnn2_open cnn2_static6 cnn2_full; do run ${cell} 33_${cell}_s1 train.seed=1; done
run cnn2_stim8 34_cnn2_stim8_s1 train.seed=1
run cnn_stim8  35_cnn_stim8_s1  train.seed=1

echo; echo "elapsed: $(( SECONDS / 60 )) min"; echo "failures:"; grep -lEi "Traceback|Error executing job" logs/sync_d/*.log || echo "  none"
