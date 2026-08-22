#!/usr/bin/env bash
set -uo pipefail

# Sort-of-CLEVR baselines. 43 runs, ~11.5 h. Every sweep lives in its
# experiment config; these are plain calls.
#
# Read train_loss/cross_entropy before any accuracy, and report accuracy
# as a delta above the per-subtype prior-optimal floors, never raw.

mkdir -p logs/soc_baselines

#  3 runs  0.0 h -- floor
python main.py task=sort_of_clevr experiment=sort_of_clevr/baselines/question_only \
  2>&1 | tee logs/soc_baselines/01_question_only.log

# 12 runs  2.3 h -- CNN+MLP over the whole question pathway
python main.py task=sort_of_clevr experiment=sort_of_clevr/baselines/cnn_mlp \
  2>&1 | tee logs/soc_baselines/02_cnn_mlp.log

#  6 runs  1.6 h -- Relation Network
python main.py task=sort_of_clevr experiment=sort_of_clevr/baselines/relnet \
  2>&1 | tee logs/soc_baselines/03_relnet.log

#  9 runs  3.3 h -- FiLM
python main.py task=sort_of_clevr experiment=sort_of_clevr/baselines/film \
  2>&1 | tee logs/soc_baselines/04_film.log

#  6 runs  1.0 h -- transformer
python main.py task=sort_of_clevr experiment=sort_of_clevr/baselines/transformer \
  2>&1 | tee logs/soc_baselines/05_transformer.log

#  6 runs  1.5 h -- syncnet, phase vs frozen
python main.py task=sort_of_clevr experiment=sort_of_clevr/baselines/syncnet_ref \
  2>&1 | tee logs/soc_baselines/06_syncnet.log

#  1 run   1.9 h -- RelNet at full spatial resolution
python main.py task=sort_of_clevr experiment=sort_of_clevr/baselines/relnet_full \
  2>&1 | tee logs/soc_baselines/07_relnet_full.log

echo; echo "failures:"
grep -lEi "Traceback|Error executing job" logs/soc_baselines/*.log || echo "  none"
