#!/usr/bin/env bash
# 48-hour SQOOP queue — single 4090, sequential, priority-ordered.
# 100k steps/run. MIN_PER_RUN is a conservative per-run estimate used by
# the deadline guard; check the first run's wall time in the log and
# restart with a corrected value if it's badly off (the guard then
# sheds low-priority blocks automatically).
#
#   tmux new -s sqoop48
#   bash run_sqoop_48h.sh 2>&1 | tee run_sqoop_48h.log
set -uo pipefail   # no -e: one failed run must not kill the queue

DEADLINE=$(( $(date +%s) + 48*3600 ))
MIN_PER_RUN=90
MARGIN=$(( MIN_PER_RUN*60 ))

guard () {
  local need_runs=$1 name=$2
  local need_s=$(( need_runs * MIN_PER_RUN * 60 + MARGIN ))
  if (( $(date +%s) + need_s > DEADLINE )); then
    echo "[queue] SKIP block '$name' (${need_runs} runs): not enough time"
    return 1
  fi
  echo "[queue] START block '$name' (${need_runs} runs) at $(date)"
}

E="experiment=sqoop/baseline_100k"
T="task=sqoop"

# ---------------------------------------------------------------- P0
# Datasets: one per #rhs, ~100k train examples each (num_repeats scaled
# so dataset size stays constant across rhs, as in Bahdanau et al.).
# Idempotent; hard-fails if any prepare fails.
declare -A REPEATS=( [1]=2778 [2]=1389 [4]=695 [8]=348 [18]=155 )
for rhs in 1 2 4 8 18; do
  if [ ! -f "data/sqoop-rhs${rhs}/train.npz" ]; then
    echo "[queue] preparing sqoop rhs=${rhs}..."
    python prepare_dataset.py task=sqoop \
      dataset.rhs_variety=${rhs} dataset.num_repeats=${REPEATS[$rhs]} || {
        echo "[queue] FATAL: prepare failed for rhs=${rhs}"; exit 1; }
  else
    echo "[queue] dataset sqoop-rhs${rhs} found, skipping prepare"
  fi
done

# ---------------------------------------------------------------- P1 (9)
# Model comparison at rhs=1 -- the hardest split, where routing must
# earn its keep. conv_lstm is the no-routing floor.
guard 9 "models_at_rhs1" && \
python main.py -m $T $E \
  model=sqoop/recurrent_syncnet,sqoop/syncnet_v3,sqoop/conv_lstm \
  dataset.rhs_variety=1 train.seed=0,1,2 \
  wandb.group=sqoop48_models_rhs1

# ---------------------------------------------------------------- P2 (8)
# The #rhs curve for the recurrent model (rhs=1 covered by P1):
# test_unseen accuracy vs rhs_variety is the compositional-
# generalisation headline figure.
guard 8 "recurrent_rhs_curve" && \
python main.py -m $T $E \
  dataset.rhs_variety=2,4,8,18 train.seed=0,1 \
  wandb.group=sqoop48_recurrent_rhs

# ---------------------------------------------------------------- P3 (8)
# Same curve for the floor and the scaffolding control (1 seed each --
# these anchor the plot, they don't need tight error bars).
guard 4 "conv_lstm_rhs_curve" && \
python main.py -m $T $E model=sqoop/conv_lstm \
  dataset.rhs_variety=2,4,8,18 train.seed=0 \
  wandb.group=sqoop48_convlstm_rhs

guard 4 "v3_rhs_curve" && \
python main.py -m $T $E model=sqoop/syncnet_v3 \
  dataset.rhs_variety=2,4,8,18 train.seed=0 \
  wandb.group=sqoop48_v3_rhs

# ---------------------------------------------------------------- P4 (6)
# Gating ablations at rhs=1 (same 2x2 logic as the sort_of_clevr queue:
# dt=0 -> frozen random gates; det+dt=0 -> all-open dense messaging).
guard 2 "ablate_frozen_random" && \
python main.py -m $T $E model.dt=0.0 dataset.rhs_variety=1 \
  train.seed=0,1 wandb.group=sqoop48_ablate_frozen

guard 2 "ablate_dense_open" && \
python main.py -m $T $E model.dt=0.0 model.deterministic_phase=true \
  dataset.rhs_variety=1 train.seed=0,1 wandb.group=sqoop48_ablate_dense

guard 2 "ablate_det_dynamic" && \
python main.py -m $T $E model.deterministic_phase=true \
  dataset.rhs_variety=1 train.seed=0,1 wandb.group=sqoop48_ablate_det

echo "[queue] done at $(date)"
