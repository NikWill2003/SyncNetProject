#!/usr/bin/env bash
set -uo pipefail

# EXPLORATORY OVERNIGHT, GPU 2 -- mechanism levers on the d=6 field cell,
# all 100k, ~85 min each, 11 runs ~15.6 h. Every lever is a new default-off
# flag (defaults verified bit-identical to the current model).
#
#   step cap     bounds each bus phase to 30 deg of rotation per step: the
#                fix for spiky phase trajectories, if overshoot is real
#   field cap    same bound (45 deg) inside the oscillator field
#   stim EMA     smooths the stimulus target the phases chase
#   head taps    3 listening phases per question, concatenated at readout:
#                parallel access to individuated senders (comparisons)
#   dt matched   dt=0.25 T=16 vs the default dt=0.5 T=8: integration
#                fineness at the same phase travel budget (bus only, cheap)
#   slot both    a content term in the slot competition: the labelled
#                perception-upgrade row, prices the binding bottleneck
#
# READ: vs the field_stim6 baselines (.923/.784, .914/.761, 1-in-3
# collapse): collapse rate, ternary, closest/furthest, phase_freeze_drop.
# For taps: does binary hold at 200k-like levels without losing ternary.

BUDGET_S=$((16 * 3600)); EST_S=$((85 * 60)); mkdir -p logs/sync_d
run() { local exp=$1 stem=$2; shift 2
  if (( SECONDS + EST_S > BUDGET_S )); then echo "[skip] ${stem}"; return 0; fi
  python main.py task=sort_of_clevr experiment=sort_of_clevr/sync_d/${exp} "$@" 2>&1 | tee logs/sync_d/${stem}.log; }

run field_stim6 50_cap30_s0        train.seed=0 model.phase_step_max_deg=30 'wandb.tags=[sync_d,explore,cap]'
run field_stim6 51_taps3_s0        train.seed=0 model.head_taps=3 'wandb.tags=[sync_d,explore,taps]'
run field_stim6 52_fieldcap45_s0   train.seed=0 model.field_step_max_deg=45 'wandb.tags=[sync_d,explore,fieldcap]'
run field_stim6 53_slotboth_s0     train.seed=0 model.slot_read=both 'wandb.tags=[sync_d,explore,slotboth]'
run field_stim6 54_ema8_s0         train.seed=0 model.stim_ema=0.8 'wandb.tags=[sync_d,explore,ema]'
run field_stim6 55_dt25T16_s0      train.seed=0 model.dt=0.25 model.T=16 'wandb.tags=[sync_d,explore,dt]'
run field_stim6 56_cap30_s1        train.seed=1 model.phase_step_max_deg=30 'wandb.tags=[sync_d,explore,cap]'
run field_stim6 57_taps3_s1        train.seed=1 model.head_taps=3 'wandb.tags=[sync_d,explore,taps]'
run field_stim6 58_slotboth_s1     train.seed=1 model.slot_read=both 'wandb.tags=[sync_d,explore,slotboth]'
run field_stim6 59_cap30_s2        train.seed=2 model.phase_step_max_deg=30 'wandb.tags=[sync_d,explore,cap]'
run field_stim6 60_taps3_s2        train.seed=2 model.head_taps=3 'wandb.tags=[sync_d,explore,taps]'

echo; echo "elapsed: $(( SECONDS / 60 )) min"; echo "failures:"; grep -lEi "Traceback|Error executing job" logs/sync_d/*.log || echo "  none"
