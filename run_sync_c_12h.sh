#!/usr/bin/env bash
set -uo pipefail

# SYNC SCREEN C, 12-HOUR CUT -- BusNet on the standard Sort-of-CLEVR task.
# 8 runs at ~85 min each (measured), ~11.3 h: the dimension sweep first,
# then the cells that interpret it. A time guard skips any run that could not finish inside the
# budget; the optional tail after it only starts if time is left.
#
# Assumes busnet_bus job 0 (circle, phase; test .65 ternary-floor plateau)
# has finished from the earlier invocation. Cells that are a subset of a
# yaml sweep are launched as single runs with hydra.mode=RUN plus the value
# (a plain model.<key>= override does not narrow a yaml-defined sweep, and
# the sweeper's dotted keys cannot be overridden from the command line).
# Single runs land in outputs/<dataset>/<date>/<time> rather than multirun/.
#
#   1-5  d = 2, 3, 4, 6, 8   the dimension sweep (one multirun call, ~7 h):
#                            d=2 is the circle on the vector implementation
#                            (should match the finished .65 run), 3 is the
#                            Coalitions fix, 6 is one axis per object
#   6    d=6 static          capacity (fixed learned axes) vs dynamics
#   7    d=6 + stimulus      the combination most likely to work
#   8    channels/attn       per-sender ceiling on this architecture
#
# READ: eval_model/head_own_align vs head_other_align, n_clusters_eff,
# head_tvar, test_interventions/phase_zero_drop, binary / ternary accuracy
# against .49 (question-only floor) and channels/attn (ceiling).

BUDGET_S=$((12 * 3600))
EST_S=$((85 * 60))
SWEEP5_S=$((5 * 85 * 60))
mkdir -p logs/sync_c

run() {  # run <experiment> <log-stem> [extra overrides...]
  local exp="$1" stem="$2"; shift 2
  local need=$EST_S
  [[ "$exp" == busnet_dim ]] && need=$SWEEP5_S      # five-run multirun
  if (( SECONDS + need > BUDGET_S )); then
    echo "[skip] ${exp} ${*:-}: $(( (BUDGET_S - SECONDS) / 60 )) min left, need ~$(( need / 60 ))"
    return 0
  fi
  python main.py task=sort_of_clevr experiment=sort_of_clevr/sync_c/${exp} "$@" \
    2>&1 | tee logs/sync_c/${stem}.log
}

run busnet_dim          01_busnet_dim
run busnet_dim_static   02_busnet_dim_static
run busnet_dim_stimulus 03_busnet_dim_stimulus_6  hydra.mode=RUN model.osc_dim=6
run busnet_channels     04_busnet_channels

# ---- capacity controls: is the d-gain orientation coding or raw bandwidth? ----
run busnet_capacity_rx1     09_busnet_capacity_rx1
run busnet_capacity_width   10_busnet_capacity_width
run busnet_capacity_static4 11_busnet_capacity_static4

# ---- pixels: colour-keyed slots on a CNN, the comparable version ----
run busnet_pixels_bus      12_busnet_pixels_bus
run busnet_pixels_channels 13_busnet_pixels_channels

# ---- pixels, no identities: slot attention, the fair 'image in' version ----
run busnet_slots_bus      14_busnet_slots_bus
run busnet_slots_channels 15_busnet_slots_channels

# ---- all-synchrony: oscillator field + phase slot attention + bus ----
run busnet_field_bus      16_busnet_field_bus
run busnet_field_control  17_busnet_field_control
run busnet_field_channels 18_busnet_field_channels

# ---- optional tail: only if the guard finds time ----
run busnet_bus          05_busnet_bus_open        hydra.mode=RUN model.bus_phase=open
run busnet_stimulus     06_busnet_stimulus
run busnet_circle_omega 07_busnet_circle_omega
run busnet_dim_stimulus 08_busnet_dim_stimulus_3  hydra.mode=RUN model.osc_dim=3
run busnet_d6_omega     09_busnet_d6_omega
run busnet_circle_T16   10_busnet_circle_T16
run busnet_bus          11_busnet_bus_zero        hydra.mode=RUN model.bus_phase=zero

echo; echo "elapsed: $(( SECONDS / 60 )) min"; echo "failures:"
grep -lEi "Traceback|Error executing job" logs/sync_c/*.log || echo "  none"
