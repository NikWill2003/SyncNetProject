#!/usr/bin/env bash
set -uo pipefail

# SQOOP baselines. 46 runs, ~11.5 h. Every sweep lives in its experiment
# config; these are plain calls. Uses the rhs=18 dataset throughout.
#
# READ train_loss/cross_entropy FIRST. ln2 = 0.69315 means the run never
# left chance and its accuracy describes nothing. Roughly 40% of SQOOP
# runs do this and escape depends on the seed, so aggregate conditionally.

mkdir -p logs/sqoop_baselines

#  3 runs  0.0 h -- data check, must read 0.500 exactly
python main.py task=sqoop experiment=sqoop/baselines/question_only \
  2>&1 | tee logs/sqoop_baselines/01_question_only.log

# 12 runs  3.0 h -- Conv+LSTM over the whole question pathway
python main.py task=sqoop experiment=sqoop/baselines/conv_lstm_grid \
  2>&1 | tee logs/sqoop_baselines/02_conv_lstm_grid.log

#  6 runs  1.4 h -- Relation Network
python main.py task=sqoop experiment=sqoop/baselines/relnet \
  2>&1 | tee logs/sqoop_baselines/03_relnet.log

#  9 runs  2.8 h -- FiLM
python main.py task=sqoop experiment=sqoop/baselines/film \
  2>&1 | tee logs/sqoop_baselines/04_film.log

# 10 runs  1.3 h -- transformer escape rate, ten seeds at one config
python main.py task=sqoop experiment=sqoop/baselines/transformer_escape \
  2>&1 | tee logs/sqoop_baselines/05_transformer_escape.log

#  5 runs  1.2 h -- syncnet
python main.py task=sqoop experiment=sqoop/baselines/syncnet_ref \
  2>&1 | tee logs/sqoop_baselines/06_syncnet.log

#  1 run   1.8 h -- RelNet at full spatial resolution
python main.py task=sqoop experiment=sqoop/baselines/relnet_full \
  2>&1 | tee logs/sqoop_baselines/07_relnet_full.log

echo; echo "failures:"
grep -lEi "Traceback|Error executing job" logs/sqoop_baselines/*.log || echo "  none"
