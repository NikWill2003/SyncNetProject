#!/usr/bin/env bash
set -uo pipefail

# SYNC SCREEN A -- the existing SyncNet made competent and expressible,
# then pushed on object tokens (no patchify), stacked fixes ("v3"), and a
# Kuramoto regime map. 108 runs, ~37 h on one 4090 at ~20 min/run
# (extrapolated from the 15-min canonical run). 100k steps, Sort-of-CLEVR.
# One seed everywhere except the three anchor cells, which get seed 1 as
# well so the cells everything else is compared against carry a noise
# estimate. Ordered most-informative-first: an overrun costs the tail.
#
# Every run carries the analysis bundle (conf/callbacks/sort_of_clevr/sync):
#   test_interventions/*   gate forced open / zero / frozen / shuffled,
#                          phases frozen / shuffled: accuracy drops overall,
#                          per family, per subtype (+ heatmap)
#   t_variance/*           accuracy vs test-time T, overall / binary / ternary
#   test_binding/*         what modules read (object coverage, purity,
#                          queried-object attention), coalition score,
#                          R(t), gate entropy(t), effective clusters
#                          (+ gate heatmaps by family, attention maps,
#                          phase trajectories, module x object matrix)
#
# READ, per run: gate_zero_drop and gate_open_drop first (were messages
# used; was selectivity used); read_overlap / obj_coverage (segregation);
# phase_R and gate_entropy (collapse to global sync / uniform gating);
# coalition_score on objects runs (does the gate form the question's
# graph); then ternary accuracy as a delta above the 0.538 floor.
# eval_model/obj_found must read 6.0 on every objects run.

mkdir -p logs/sync_a

run() {  # run <experiment> <log-stem> [extra overrides...]
  local exp="$1" stem="$2"; shift 2
  python main.py task=sort_of_clevr experiment=sort_of_clevr/sync_a/${exp} "$@" \
    2>&1 | tee logs/sync_a/${stem}.log
}

#  5 runs  1.7 h -- competence ladder: canonical -> cnn -> +heads -> +content -> v2
for rung in 0_canonical 1_cnn 2_heads 3_content 4_v2; do
  run ladder_${rung} 01_ladder_${rung}
done

#  7 runs  2.3 h -- gate null on v2, + zero (no-comm) and phase_io (directed)
run gate_null_v2 02_gate_null_v2

# 11 runs  3.7 h -- NO PATCHIFY: object tokens, one module per object
run objects_partition          03_objects_partition
run objects_partition_express  04_objects_partition_express
run objects_io_sharp           05_objects_io_sharp
run objects_free               06_objects_free

#  6 runs  2.0 h -- segregation without the quadrant crutch
run segregation_v2 07_segregation_v2

# 12 runs  4.0 h -- v3: every cheap fix stacked, gate null on grid and on objects
run v3_quadrant 08_v3_quadrant
run v3_objects  09_v3_objects

# 13 runs  4.3 h -- seed 1 for the anchor cells
run ladder_4_v2       10_ladder_4_v2_seed1       train.seed=1
run gate_null_v2      11_gate_null_v2_seed1      train.seed=1
run objects_partition 12_objects_partition_seed1 train.seed=1

#  7 runs  2.3 h -- can the gate close: sharpening, S^{d-1}, zero diagonal
run express_sharpen  13_express_sharpen
run express_dim      14_express_dim
run express_zerodiag 15_express_zerodiag

#  8 runs  2.7 h -- make open cost something, on the grid and on objects
run pressure_agg          16_pressure_agg
run pressure_topk         17_pressure_topk
run objects_pressure_agg  18_objects_pressure_agg
run objects_pressure_topk 19_objects_pressure_topk

# 19 runs  6.3 h -- dynamics regime, on the grid and on objects
for cell in init dt omega coupling drive T; do
  run dyn_${cell} 20_dyn_${cell}
done
for cell in dt init coupling T; do
  run objects_dyn_${cell} 21_objects_dyn_${cell}
done

#  9 runs  3.0 h -- Kuramoto regime map: dt x omega spread, omega fixed
run regime_map 22_regime_map

# 10 runs  3.3 h -- v3 free modules, conditioning, surplus modules, sync readout
run segregation_v3 23_segregation_v3
run conditioning_v2 24_conditioning_v2
run objects_free_M 25_objects_free_M
run sync_readout   26_sync_readout
run scale_free_v2  27_scale_free_v2

#  1 run   0.7 h -- v2 at 200k steps
run ladder_5_v2_200k 28_ladder_5_v2_200k

echo; echo "failures:"
grep -lEi "Traceback|Error executing job" logs/sync_a/*.log || echo "  none"
