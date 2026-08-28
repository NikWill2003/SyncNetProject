#!/usr/bin/env bash
set -uo pipefail

# EXPLORATORY OVERNIGHT, GPU 1 -- the scaled configuration's neighbourhood.
# ~15 h. The scaled cell (M=8, d=8) looked like the lock-in candidate
# mid-run; tonight answers what it needs before it can be one: its collapse
# rate at 100k (two more seeds), its same-capacity static control at 200k,
# and the schedule question (does a long flat middle let the d=6 model
# serve both question families, or is the 200k sum-code basin schedule-
# independent?).
#
# READ: scaled 100k seeds -- obj_coverage per seed (collapse rate at M=8
# vs the 1-in-3 at M=6); scaled_static 200k vs the scaled 200k run
# (computed vs stored addresses at the new capacity); the WSD run's binary
# vs ternary trajectory against the 200k cosine run's trade.

BUDGET_S=$((16 * 3600)); mkdir -p logs/sync_d
run() { local est=$1 exp=$2 stem=$3; shift 3
  if (( SECONDS + est > BUDGET_S )); then echo "[skip] ${exp} ${*:-}"; return 0; fi
  python main.py task=sort_of_clevr experiment=sort_of_clevr/sync_d/${exp} "$@" 2>&1 | tee logs/sync_d/${stem}.log; }

#  scaled collapse rate: 2 seeds at 100k (~2.9 h each)
run $((175*60)) scaled 40_scaled_100k_s1 train.seed=1 'wandb.tags=[sync_d,explore]'
run $((175*60)) scaled 41_scaled_100k_s2 train.seed=2 'wandb.tags=[sync_d,explore]'

#  the control the scaled cell needs: static addresses at the same capacity, 200k (~5.7 h)
run $((345*60)) scaled_static 42_scaled_static_200k_s0 train.seed=0 train.n_steps=200000 'wandb.tags=[sync_d,explore,200k]'

#  the schedule probe: d=6 stim at 200k under warmup-stable-decay (~2.9 h)
run $((175*60)) field_stim6 43_stim6_200k_wsd_s0 train.seed=0 train.n_steps=200000 \
  optim.lr_scheduler=warmup_stable_decay 'optim.lr_scheduler_params={warmup_steps:2000,decay_frac:0.25}' \
  'wandb.tags=[sync_d,explore,200k,wsd]'

echo; echo "elapsed: $(( SECONDS / 60 )) min"; echo "failures:"; grep -lEi "Traceback|Error executing job" logs/sync_d/*.log || echo "  none"
