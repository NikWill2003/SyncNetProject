#!/usr/bin/env bash
set -uo pipefail
# BATCH A -- SQOOP.  39 runs, ~8.8 h.  Runs start to finish unattended.
#
# No conditionals: every step runs regardless of what the previous one
# found. set -u without -e is deliberate -- a failed step must not abandon
# the rest overnight. Each step tees to logs/batch_A/, and the last line
# tells you how to find failures.
#
# Reading order still matters even though execution order does not:
#   1  is the "floor" actually a floor?      (mislabels every comparison)
#   2  what is the ceiling with no systematic demand?
#   3  did the transformer ever leave chance? (if not, 7 measures nothing)
#   then 4-7.
#
# Datasets: rhs=18 exists. rhs 1,2,4,8 and 35 are built on first use by
# build_dataloaders, ~40 min each, ~0.6 GB each. That is inside the 8.8 h
# only if they already exist -- add up to ~3.2 h if not.

mkdir -p logs/batch_A

# 1. CONV+LSTM ABLATION                                     4 runs  0.6 h
# Ours 0.41% error at rhs=18; Bahdanau's off the top of a 14% axis. Two
# readout choices in our port hand the model absolute position, and
# left_of/above are pure position predicates.
python main.py task=sqoop experiment=sqoop/baselines/conv_lstm_ablation \
  2>&1 | tee logs/batch_A/01_conv_lstm_ablation.log

# 2. IID CONTROL (rhs=35, nothing held out)                 3 runs  0.6 h
# Pins the ceiling so every rhs point below reads as a delta from it.
python main.py task=sqoop experiment=sqoop/baselines/iid_control model=sqoop/conv_lstm \
  2>&1 | tee logs/batch_A/02a_iid_conv_lstm.log
python main.py task=sqoop experiment=sqoop/baselines/iid_control model=sqoop/syncnet \
  2>&1 | tee logs/batch_A/02b_iid_syncnet.log
python main.py task=sqoop experiment=sqoop/baselines/iid_control model=sqoop/transformer \
  2>&1 | tee logs/batch_A/02c_iid_transformer.log

# 3. TRANSFORMER DIAGNOSTIC                                 8 runs  2.0 h
# All 16 arms finished at ln2 with train acc 0.50. Not a wiring fault --
# the same model overfits 512 examples to 0.042 in 1000 steps. Read
# train_loss/cross_entropy, not accuracy.
python main.py task=sqoop experiment=sqoop/diagnostics/transformer_lr \
  2>&1 | tee logs/batch_A/03_transformer_lr.log

# 4. COMMUNICATION RERUN                                   10 runs  2.5 h
# Died after its first cell last time. Gives phase-vs-frozen on a second
# dataset and a like-for-like none-vs-quadrant at matched readout=sum.
python main.py task=sqoop experiment=sqoop/syncnet/communication \
  2>&1 | tee logs/batch_A/04_communication.log

# 5. CONV+LSTM RHS CURVE                                    4 runs  0.6 h
# The floor at each rhs. Without it the other two curves have nothing to
# be a delta above.
for RHS in 1 2 4 8; do
  python main.py task=sqoop experiment=sqoop/baselines/conv_lstm \
    dataset.rhs_variety=$RHS model.encoder.ch=128 hydra.mode=RUN \
    2>&1 | tee logs/batch_A/05_conv_lstm_rhs$RHS.log
done

# 6. SYNCNET RHS CURVE                                      5 runs  1.2 h
python main.py task=sqoop experiment=sqoop/syncnet/rhs \
  2>&1 | tee logs/batch_A/06_syncnet_rhs.log

# 7. TRANSFORMER RHS CURVE                                  5 runs  1.3 h
# Runs unconditionally. If step 3 left every arm at ln2 this curve is
# drawn through a model stuck at chance -- discard it, do not debug it.
python main.py task=sqoop experiment=sqoop/baselines/transformer_rhs \
  2>&1 | tee logs/batch_A/07_transformer_rhs.log

echo
echo "batch A done. failures:"
grep -lEi "Traceback|Error executing job" logs/batch_A/*.log || echo "  none"
