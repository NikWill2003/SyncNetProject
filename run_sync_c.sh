#!/usr/bin/env bash
set -uo pipefail

# SYNC SCREEN C -- BusNet on the standard Sort-of-CLEVR task. 22 runs at
# ~85 min each (measured), ~31 h on one 4090. Same recipe as sync_a / sync_b.
#
# The first run (circle, phase) plateaued at .65: the bus aggregates
# (count_same_shape .98) and addresses one bit (query_shape .99) but never
# moves a position to the head, and the relational questions stay at the
# floor, with R = .93 (open-bus behaviour). Two failures, two causes: the
# position channel is a discovery problem the circle has room for (the
# stimulus drive is the test), and the relational questions need five or
# six senders at the head, which one wire can only do by time-sharing
# (two circle cells test that) or by a medium with more axes (the d-sweep,
# and the remaining knobs moved to d=6, one axis per object).
#
# READ: eval_model/head_own_align vs head_other_align, n_clusters_eff,
# head_tvar (sweeping), test_interventions/phase_zero_drop, then binary /
# ternary accuracy against bus_phase=zero (floor) and channels/attn
# (ceiling). eval_model/obj_found must read 6.0.
#
# The first invocation already completed busnet_bus job 0 (phase). Hydra
# multirun does not resume; on a restart pass `model.bus_phase=open,zero`
# to busnet_bus to skip it -- launch the two as single runs instead:
#   run busnet_bus 01_open hydra.mode=RUN model.bus_phase=open
#   run busnet_bus 01_zero hydra.mode=RUN model.bus_phase=zero
# (a plain model.<key>= override does not narrow a yaml-defined sweep).

mkdir -p logs/sync_c

run() {  # run <experiment> <log-stem> [extra overrides...]
  local exp="$1" stem="$2"; shift 2
  python main.py task=sort_of_clevr experiment=sort_of_clevr/sync_c/${exp} "$@" \
    2>&1 | tee logs/sync_c/${stem}.log
}

#  4 runs  5.7 h -- the medium on the circle: phase / open / silent, and per-sender attention
run busnet_bus      01_busnet_bus
run busnet_channels 02_busnet_channels

#  1 run   1.4 h -- circle + stimulus: can the queried object leave the crowd's axis
run busnet_stimulus 03_busnet_stimulus

#  8 runs 11.3 h -- the dimension of the medium: d-sweep, static addresses, d + stimulus
run busnet_dim          04_busnet_dim
run busnet_dim_static   05_busnet_dim_static
run busnet_dim_stimulus 06_busnet_dim_stimulus

#  2 runs  2.8 h -- circle time-division: fixed omega spread, sixteen steps
run busnet_circle_omega 07_busnet_circle_omega
run busnet_circle_T16   08_busnet_circle_T16

#  5 runs  7.1 h -- d=6: formation (fixed omega, learned theta0), steps, width
run busnet_d6_omega         09_busnet_d6_omega
run busnet_d6_learned_theta 10_busnet_d6_learned_theta
run busnet_d6_T             11_busnet_d6_T
run busnet_d6_width         12_busnet_d6_width

echo; echo "failures:"
grep -lEi "Traceback|Error executing job" logs/sync_c/*.log || echo "  none"
