#!/usr/bin/env bash
set -uo pipefail

# SYNC SCREEN B -- new models: where should the phase live?
# 106 runs, ~39 h on one 4090 at ~22 min/run. 100k steps, Sort-of-CLEVR.
# Independent of run_sync_a.sh: run them on the two GPUs at once. One
# seed everywhere except the three anchor cells (pb_full, pb_objects,
# osc_field), which get seed 1 as well. Ordered most-informative-first.
#
# PhaseBind: tokens and modules are one oscillator system; the read is
# phase-gated with competition over modules; the message gate is the same
# phase variable. OscField: an AKOrN-style oscillator field with no
# modules, read out by "find the object by content, read what is in
# phase with it". Both run on the CNN grid and on object tokens.
#
# Every run carries the analysis bundle (see run_sync_a.sh). READ, per
# PhaseBind run: test_binding/obj_coverage, module_purity, queried_attn
# (did modules take objects), eval_model/assign_purity and module_use,
# bind_R (tokens a module reads are in phase), within_obj_R vs
# between_obj_align (binding by synchrony at the token level), then the
# interventions: tok_shuffle_drop and lambda0_drop are the
# binding-by-phase tests, gate_zero_drop the messaging test. For OscField:
# lambda0_drop and phase_shuffle_drop against the content-readout cell.
# eval_model/obj_found must read 6.0 on every objects run.

mkdir -p logs/sync_b

run() {  # run <experiment> <log-stem> [extra overrides...]
  local exp="$1" stem="$2"; shift 2
  python main.py task=sort_of_clevr experiment=sort_of_clevr/sync_b/${exp} "$@" \
    2>&1 | tee logs/sync_b/${stem}.log
}

#  7 runs  2.6 h -- the full model on the grid, then on object tokens
run pb_full    01_pb_full
run pb_objects 02_pb_objects

#  6 runs  2.2 h -- what the phase does at the read; competition; hard assignment
run pb_read 03_pb_read
run pb_norm 04_pb_norm
run pb_hard 05_pb_hard

#  3 runs  1.1 h -- the message gate: open / zero / attention
run pb_gate 06_pb_gate

#  6 runs  2.2 h -- remove one coupling path at a time; AKOrN stimulus
for cell in coupling_tok coupling_modtok coupling_modmod stimulus; do
  run pb_${cell} 07_pb_${cell}
done

#  4 runs  1.3 h -- oscillator field, no modules: phase used x oscillators interact
run osc_field 08_osc_field

# 11 runs  4.0 h -- seed 1 for the anchor cells
run pb_full    09_pb_full_seed1    train.seed=1
run pb_objects 10_pb_objects_seed1 train.seed=1
run osc_field  11_osc_field_seed1  train.seed=1

# 11 runs  4.0 h -- objects: read variants, coupling paths, module coupling
run pb_objects_read     12_pb_objects_read
run pb_objects_readonly 13_pb_objects_readonly
run pb_objects_coupling 14_pb_objects_coupling
run pb_objects_modmod   15_pb_objects_modmod

#  3 runs  1.1 h -- quadrant control, and is the gate load-bearing under it
run pb_quadrant      16_pb_quadrant
run pb_quadrant_gate 17_pb_quadrant_gate

# 10 runs  3.7 h -- gate shape, aggregation, phase weight in the read
run pb_gate_shape 18_pb_gate_shape
run pb_msg        19_pb_msg
run pb_lambda     20_pb_lambda

# 18 runs  6.6 h -- dimension x steps, step size x init, M x readout, frequencies
for cell in dyn dt size omega; do
  run pb_${cell} 21_pb_${cell}
done

#  8 runs  2.9 h -- objects: dynamics, step size, surplus modules
run pb_objects_dyn 22_pb_objects_dyn
run pb_objects_dt  23_pb_objects_dt
run pb_objects_M   24_pb_objects_M

#  2 runs  0.7 h -- encoder and conditioning confounds on the new model
run pb_encoder 25_pb_encoder
run pb_cond    26_pb_cond

# 17 runs  6.2 h -- oscillator field variants
for cell in res dyn long d2 slots width kernel dt static res_content; do
  run osc_field_${cell} 27_osc_field_${cell}
done

echo; echo "failures:"
grep -lEi "Traceback|Error executing job" logs/sync_b/*.log || echo "  none"
