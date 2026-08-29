#!/usr/bin/env bash
set -uo pipefail
# QUICK PROBE: the field-bus SyncNet on SQOOP at rhs=18, 50k steps, 5 seeds.
# Half the seeded budget, so this measures the ESCAPE RATE BY 50k -- a lower
# bound on the 100k rate (the cosine decays twice as fast here, so a seed
# that misses at 50k can still fire at 100k). Tagged probe50k so it can
# never be averaged into the rhs_seeded table. ~15-17 min per run compiled,
# ~80-90 min total on one 4090.
#
# READ: train CE detaching from 0.693 and eval accuracy leaving .50 = the
# transition fired; read_entropy falling from 1.0 = slots specialising.
# Compare escapes against the gated SyncNet (25/25 by 100k) and the
# transformer (~0/25 by 100k).
mkdir -p logs/sqoop
python main.py task=sqoop experiment=sqoop/busnet/stim \
  dataset.rhs_variety=18 'train.seed=0,1,2,3,4' train.n_steps=50000 \
  'wandb.tags=[sqoop_busnet,probe50k]' \
  2>&1 | tee logs/sqoop/busnet_stim_rhs18_probe50k_s0-4.log
