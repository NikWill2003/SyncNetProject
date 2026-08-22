#!/usr/bin/env bash
set -uo pipefail

# Sort-of-CLEVR baselines. 43 runs, ~11.5 h. Unattended, no conditionals.
#
# Read train_loss/cross_entropy before any accuracy, and report accuracy
# as a delta above the per-subtype prior-optimal floors, never raw.
# 3 seeds throughout, which is a screen; 5 for anything stated as a claim.

mkdir -p logs/soc_baselines

# 1. FLOOR                                            3 runs   0.0 h
# Not a model result: how much the question alone carries, per subtype.
python main.py task=sort_of_clevr experiment=sort_of_clevr/baselines/question_only \
  'hydra.sweeper.params.train.seed=0,1,2' \
  2>&1 | tee logs/soc_baselines/01_question_only.log

# 2. CNN+MLP, QUESTION PATHWAY SWEEP                 12 runs   2.3 h
# q_pool x fusion. The two published baselines are opposite corners of
# this grid -- CNN+MLP is mlp/readout, Conv+LSTM is lstm/spatial -- so the
# off-diagonal cells turn the question pathway into a measured variable
# rather than a confound between the two tasks' baselines.
# Expect mlp/readout to solve non-relational and plateau on relational.
python main.py task=sort_of_clevr experiment=sort_of_clevr/baselines/cnn_mlp \
  'hydra.sweeper.params.model.q_pool=mlp,lstm' \
  'hydra.sweeper.params.model.fusion=readout,spatial' \
  'hydra.sweeper.params.train.seed=0,1,2' \
  2>&1 | tee logs/soc_baselines/02_cnn_mlp.log

# 3. RELATION NETWORK                                 6 runs   1.6 h
# The model this dataset was introduced with. Target: above 94% on both
# families where CNN+MLP plateaus near 63% relational. Reproducing that
# gap validates the whole pipeline. Cost goes as pair_spatial^4.
python main.py task=sort_of_clevr experiment=sort_of_clevr/baselines/relnet \
  'hydra.sweeper.params.model.pair_spatial=3,5' \
  'hydra.sweeper.params.train.seed=0,1,2' \
  2>&1 | tee logs/soc_baselines/03_relnet.log

# 4. FILM                                             9 runs   3.3 h
# Per-block modulation from a question generator. NOT the same as the
# q_conditioning=film axis in step 5 -- do not share a label in a table.
python main.py task=sort_of_clevr experiment=sort_of_clevr/baselines/film \
  'hydra.sweeper.params.model.n_blocks=2,4,6' \
  'hydra.sweeper.params.train.seed=0,1,2' \
  2>&1 | tee logs/soc_baselines/04_film.log

# 5. TRANSFORMER                                      6 runs   1.0 h
# Matched-capacity attention comparison, at patch 5 (225 tokens). Patch 15
# collapses ternary from 0.867 to 0.573, so it measures a dead regime.
python main.py task=sort_of_clevr experiment=sort_of_clevr/baselines/transformer_conditioning \
  'hydra.sweeper.params.model.q_conditioning=film,broadcast_cat' \
  'hydra.sweeper.params.model.patch_size=5' \
  'hydra.sweeper.params.train.seed=0,1,2' \
  2>&1 | tee logs/soc_baselines/05_transformer.log

# 6. SYNCNET, PHASE vs FROZEN                         6 runs   1.5 h
# The model under test, in the same table as the baselines. Frozen shares
# parameter count, gate function and message pathway and differs only in
# whether the phases evolve.
python main.py task=sort_of_clevr experiment=sort_of_clevr/syncnet/gate_null \
  'hydra.sweeper.params.model.gate_mode=phase,frozen' \
  'hydra.sweeper.params.train.seed=0,1,2' \
  2>&1 | tee logs/soc_baselines/06_syncnet.log

# 7. RELNET AT FULL RESOLUTION                        1 run    1.9 h
# pair_spatial=8 is 4096 pairs and 5x the cost of 5. One seed, to check
# whether the coarse grid is what limits RelNet here.
python main.py task=sort_of_clevr experiment=sort_of_clevr/baselines/relnet \
  'hydra.sweeper.params.model.pair_spatial=8' \
  'hydra.sweeper.params.train.seed=0' \
  2>&1 | tee logs/soc_baselines/07_relnet_full.log

echo; echo "failures:"
grep -lEi "Traceback|Error executing job" logs/soc_baselines/*.log || echo "  none"
