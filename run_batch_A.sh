#!/usr/bin/env bash
set -uo pipefail
# BATCH A -- SORT-OF-CLEVR, capacity.  27 runs, ~19.2 h. Unattended.
#
# The question: the syncnet tops out at ternary ~0.63 (+0.087 over the
# 0.545 floor) while a 0.58M transformer reaches 0.867. Is that a size
# gap or an architectural one? Every arm here is the same axis on both
# families, in the working regime (patch 5).

mkdir -p logs/batch_A

# 1. TRANSFORMER LADDER                 6 runs  13.2 h
# hidden {128,256,384} x layers {4,8}, 0.58M -> 9.7M. Watch for the peak:
# a 10.8M transformer previously scored WORSE than 3.3M at late `token`
# conditioning, and if that repeats at film it is a conditioning result.
python main.py task=sort_of_clevr experiment=sort_of_clevr/baselines/transformer_scale \
  2>&1 | tee logs/batch_A/01_transformer_scale.log

# 2. SYNCNET WIDTH LADDER              12 runs   4.0 h
# module_dim {128,256,512,768} x 3 seeds at quadrant, 0.33M -> 1.7M+.
# quadrant PINS n_modules=4, so width is the only axis available here.
python main.py task=sort_of_clevr experiment=sort_of_clevr/syncnet/scale \
  2>&1 | tee logs/batch_A/02_syncnet_scale.log

# 3. SYNCNET MODULE-COUNT LADDER        9 runs   1.9 h
# n_modules {4,8,16} x 3 seeds, which needs partition=none and therefore
# gives up the partition's +0.082. Read as "does module count help at
# all", and as a control: if accuracy tracks parameters rather than
# module structure, the modularity framing does no work.
python main.py task=sort_of_clevr experiment=sort_of_clevr/syncnet/scale_free \
  2>&1 | tee logs/batch_A/03_syncnet_scale_free.log

echo
echo "batch A done. failures:"
grep -lEi "Traceback|Error executing job" logs/batch_A/*.log || echo "  none"
