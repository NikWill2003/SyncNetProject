#!/usr/bin/env bash
set -uo pipefail
# BATCH B -- SQOOP, capacity + the remaining gaps.  55 runs, ~21.0 h.
#
# The rhs curve says conv_lstm beats the syncnet at every rhs with no
# crossover. Before that is written as architectural it has to survive
# two objections: the syncnet was smaller (0.39M vs 0.89M), and the curve
# is one run per point.

mkdir -p logs/batch_B

# 1. IID CONTROL, CORRECTED             1 run   0.2 h
# The first one ran the syncnet at its DEFAULT config (partition=none),
# so 0.513 at rhs=35 measured the weak arm, not the ceiling. My error.
python main.py task=sqoop experiment=sqoop/baselines/iid_control_quadrant \
  2>&1 | tee logs/batch_B/01_iid_quadrant.log

# 2. COMMUNICATION RERUN               10 runs   2.5 h
# Died after its first cell twice now. The only missing piece of the
# phase-vs-frozen replication on a second dataset.
python main.py task=sqoop experiment=sqoop/syncnet/communication \
  2>&1 | tee logs/batch_B/02_communication.log

# 3. SYNCNET CAPACITY                   8 runs   3.1 h
# module_dim {128,256,512,768} at rhs 4 and 18 -- widest gap and best
# case. 0.39M -> 3.2M. If that does not close it, size is not why.
python main.py task=sqoop experiment=sqoop/syncnet/scale \
  2>&1 | tee logs/batch_B/03_syncnet_scale.log

# 4. TRANSFORMER CAPACITY               6 runs   3.4 h
# Stuck at ln2 at every rhs 1-18 but reaches 0.673 at rhs=35, so not
# broken -- trapped. Does 9.7M escape the basin?
python main.py task=sqoop experiment=sqoop/baselines/transformer_scale \
  2>&1 | tee logs/batch_B/04_transformer_scale.log

# 5. SYNCNET RHS CURVE, 3 SEEDS        15 runs   5.6 h
python main.py task=sqoop experiment=sqoop/syncnet/rhs_seeded \
  2>&1 | tee logs/batch_B/05_syncnet_rhs_seeded.log

# 6. CONV+LSTM RHS CURVE, 3 SEEDS      15 runs   2.2 h
# The floor at each point, with error bars. Without it the syncnet curve
# has nothing to be a delta above.
python main.py task=sqoop experiment=sqoop/baselines/rhs_seeded \
  2>&1 | tee logs/batch_B/06_conv_lstm_rhs_seeded.log

echo
echo "batch B done. failures:"
grep -lEi "Traceback|Error executing job" logs/batch_B/*.log || echo "  none"
