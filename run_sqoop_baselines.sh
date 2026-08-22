#!/usr/bin/env bash
set -uo pipefail

# SQOOP baselines. 52 runs, ~13.1 h. Unattended, no conditionals.
# Uses the rhs=18 dataset throughout; no generation needed.
#
# READ train_loss/cross_entropy FIRST. ln2 = 0.69315 means the run never
# left chance and its accuracy describes nothing. Roughly 40% of SQOOP
# runs do this and escape depends on the seed, so aggregate conditionally
# -- averaging over converged and stuck runs describes neither.

mkdir -p logs/sqoop_baselines

# 1. DATA CHECK                                       3 runs   0.0 h
# Not a baseline. Labels are balanced per (x, rel, y) cell, so this must
# read 0.500 exactly. Anything else means the generator leaks and every
# number below is void.
python main.py task=sqoop experiment=sqoop/baselines/question_only \
  'hydra.sweeper.params.train.seed=0,1,2' \
  2>&1 | tee logs/sqoop_baselines/01_question_only.log

# 2. CONV+LSTM, QUESTION PATHWAY SWEEP               12 runs   3.0 h
# Same grid as the Sort-of-CLEVR step 2, same class, opposite default
# corner. Running both makes the two tasks' conv baselines comparable.
python main.py task=sqoop experiment=sqoop/baselines/conv_lstm \
  'hydra.sweeper.params.model.q_pool=mlp,lstm' \
  'hydra.sweeper.params.model.fusion=readout,spatial' \
  'hydra.sweeper.params.train.seed=0,1,2' \
  2>&1 | tee logs/sqoop_baselines/02_conv_lstm.log

# 3. RELATION NETWORK                                 6 runs   1.4 h
# Bahdanau et al. report RelNet as not training on SQOOP, so a flat run at
# ln2 is consistent with the literature rather than a broken port. Check
# train cross-entropy before concluding either way.
python main.py task=sqoop experiment=sqoop/baselines/relnet \
  'hydra.sweeper.params.model.pair_spatial=3,5' \
  'hydra.sweeper.params.train.seed=0,1,2' \
  2>&1 | tee logs/sqoop_baselines/03_relnet.log

# 4. FILM                                             9 runs   2.8 h
# The external calibration point SQOOP currently lacks: Bahdanau et al.
# report per pair-variety error rates for FiLM.
python main.py task=sqoop experiment=sqoop/baselines/film \
  'hydra.sweeper.params.model.n_blocks=2,4,6' \
  'hydra.sweeper.params.train.seed=0,1,2' \
  2>&1 | tee logs/sqoop_baselines/04_film.log

# 5. TRANSFORMER ESCAPE RATE                         10 runs   1.3 h
# Ten seeds, one configuration. The transformer is a seed lottery here --
# some seeds reach high accuracy, most sit at ln2 -- so the quantity of
# interest is the FRACTION that escape, which needs seeds not sweeps.
python main.py task=sqoop experiment=sqoop/baselines/transformer \
  'hydra.sweeper.params.train.seed=0,1,2,3,4,5,6,7,8,9' \
  2>&1 | tee logs/sqoop_baselines/05_transformer_escape.log

# 6. SYNCNET                                          5 runs   1.2 h
# The model under test, at claim-grade seed count.
python main.py task=sqoop experiment=sqoop/syncnet/rhs_seeded \
  'hydra.sweeper.params.dataset.rhs_variety=18' \
  'hydra.sweeper.params.train.seed=0,1,2,3,4' \
  2>&1 | tee logs/sqoop_baselines/06_syncnet.log

# 7. EXTRA SEEDS ON THE TWO STRONGEST BASELINES       6 runs   1.7 h
# Takes conv_lstm and film to 6 seeds at their default settings, which is
# where the escape-conditioned means need the most support.
python main.py task=sqoop experiment=sqoop/baselines/conv_lstm \
  'hydra.sweeper.params.train.seed=3,4,5' \
  2>&1 | tee logs/sqoop_baselines/07a_conv_lstm_seeds.log
python main.py task=sqoop experiment=sqoop/baselines/film \
  'hydra.sweeper.params.train.seed=3,4,5' \
  2>&1 | tee logs/sqoop_baselines/07b_film_seeds.log

# 8. RELNET AT FULL RESOLUTION                        1 run    1.8 h
# pair_spatial=8 is 4096 pairs, 5x the cost of 5. One seed.
python main.py task=sqoop experiment=sqoop/baselines/relnet \
  'hydra.sweeper.params.model.pair_spatial=8' \
  'hydra.sweeper.params.train.seed=0' \
  2>&1 | tee logs/sqoop_baselines/08_relnet_full.log

echo; echo "failures:"
grep -lEi "Traceback|Error executing job" logs/sqoop_baselines/*.log || echo "  none"
