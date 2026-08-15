#!/usr/bin/env bash
set -uo pipefail
# BATCH B -- SORT-OF-CLEVR.  34 runs, ~14.2 h.  Unattended; no conditionals.
# Independent of batch A, so run it on the other card. SOC data is ~0.6 GB
# and already built.
#
# Drop step 3 to land at 11.2 h if you want both batches even.

mkdir -p logs/batch_B

# 1. GATE NULL AT 5 SEEDS                                  25 runs  5.3 h
# The clearest result in the project, currently one seed per arm: at
# quadrant every gate_mode landed within 0.014 ternary of every other,
# frozen and open included, while the partition alone was worth +0.082.
# At sigma=0.022 one seed cannot separate 0.627 from 0.625; five gives
# ~0.010 resolution, enough to state the null rather than hint at it.
python main.py task=sort_of_clevr experiment=sort_of_clevr/syncnet/gate_null \
  2>&1 | tee logs/batch_B/01_gate_null.log

# 2. TRANSFORMER CONDITIONING AT PATCH 5                    6 runs  5.9 h
# Half its cells previously ran at patch 15, where ternary collapses from
# 0.867 to 0.573 -- a regime in which nothing can do the task.
python main.py task=sort_of_clevr experiment=sort_of_clevr/baselines/transformer_conditioning \
  'hydra.sweeper.params.model.patch_size=5' \
  2>&1 | tee logs/batch_B/02_conditioning_patch5.log

# 3. TRANSFORMER DEPTH AT PATCH 5                           3 runs  3.0 h
# I pinned this to patch 15 for cost, which capped it below the working
# regime. Stacked only (share_layer_weights=false).
python main.py task=sort_of_clevr experiment=sort_of_clevr/baselines/transformer_depth \
  'hydra.sweeper.params.model.patch_size=5' \
  'hydra.sweeper.params.model.share_layer_weights=false' \
  2>&1 | tee logs/batch_B/03_depth_patch5.log

echo
echo "batch B done. failures:"
grep -lEi "Traceback|Error executing job" logs/batch_B/*.log || echo "  none"
