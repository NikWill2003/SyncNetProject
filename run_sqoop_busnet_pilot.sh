#!/usr/bin/env bash
set -uo pipefail
# FIRST LOOK: the canonical field-bus SyncNet on SQOOP.
# One run, full rhs_seeded protocol (100k steps, 1.08M examples), so the
# number lands directly in the existing table next to conv_lstm (.999) and
# the gated SyncNet (.85) at rhs=18. Estimated ~60-75 min on one 4090.
mkdir -p logs/sqoop
python main.py task=sqoop experiment=sqoop/busnet/stim \
  dataset.rhs_variety=18 train.seed=0 \
  'wandb.tags=[sqoop_busnet,pilot]' \
  2>&1 | tee logs/sqoop/busnet_stim_rhs18_s0.log
