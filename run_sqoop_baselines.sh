#!/usr/bin/env bash
set -uo pipefail

# SQOOP baselines. Read run_sqoop_syncnet.sh only after step 2 passes.
#
# Every rhs value is its OWN dataset on disk (dir: sqoop-rhs${rhs}), so
# each has to be generated before any sweep touches it. Regeneration is
# mandatory after the label-balance fix: datasets built by the older
# generator carry a per-question class skew and step 2 will fail on them
# (correctly).

for RHS in 1 2 4 8 18; do
  python prepare_dataset.py task=sqoop dataset.rhs_variety=$RHS
done

# ---------------------------------------------------------------- 1. GATE
# 5 runs (1 seed). NOT a baseline -- a leakage test of the generator. Bayes-optimal
# question-only accuracy is exactly 0.500 by construction. STOP and fix the
# generator if any rhs converges meaningfully above that; every number
# below is uninterpretable until this reads 0.500.
python main.py task=sqoop experiment=sqoop/baselines/question_only

# ------------------------------------------------------------- 2. FLOOR
# 5 runs -- no-routing floor, the curve everything else is a delta above
python main.py task=sqoop experiment=sqoop/baselines/conv_lstm

# ------------------------------------------------- 3. CONDITIONING
# 8 runs at fixed rhs=18. Fix the winner into the two rhs configs before
# running step 4, or the curve confounds fusion with generalisation.
python main.py task=sqoop experiment=sqoop/baselines/transformer_conditioning

# ------------------------------------------------------- 4. RHS CURVE
# 5 runs -- the baseline compositionality curve, size-matched across rhs
python main.py task=sqoop experiment=sqoop/baselines/transformer_rhs
