#!/usr/bin/env bash
# 48-hour experiment queue — SyncNetProject, single 4090, sequential.
# Budget: ~30 min/run + end-of-run t_variance overhead. Blocks are ordered
# by thesis value; the deadline guard skips remaining blocks rather than
# starting one that would blow the wall clock.
#
#   nohup bash run_48h.sh > run_48h.log 2>&1 &
#
# 82 scheduled runs ≈ 41 h train + eval/viz overhead ≈ 44-45 h wall.
set -uo pipefail   # no -e: one failed run must not kill the queue

DEADLINE=$(( $(date +%s) + 48*3600 ))
MARGIN=$(( 35*60 ))   # don't start a run with <35 min left

guard () {
  local need_runs=$1 name=$2
  local need_s=$(( need_runs * 30 * 60 + MARGIN ))
  if (( $(date +%s) + need_s > DEADLINE )); then
    echo "[queue] SKIP block '$name' (${need_runs} runs): not enough time"
    return 1
  fi
  echo "[queue] START block '$name' (${need_runs} runs) at $(date)"
}

R="experiment=sort_of_clevr/recurrent_baseline"
T="task=sort_of_clevr"

# ---------------------------------------------------------------- P1 (15)
# Headline: recurrent vs v3 vs v1, 5 seeds. Everything downstream cites this.
guard 15 "headline_3way" && \
python main.py -m $T $R \
  model=sort_of_clevr/recurrent_syncnet,sort_of_clevr/syncnet_v3,sort_of_clevr/syncnet_v1 \
  train.seed=0,1,2,3,4 \
  wandb.group=48h_headline

# ---------------------------------------------------------------- P2 (15)
# Gating ablations (config-only, no code changes):
#   dt=0                        -> frozen *random* gates (dynamics removed)
#   det_phase + dt=0            -> theta_i = 0 for all i -> g == 1: dense,
#                                  ungated messaging ("no gating" control)
#   det_phase + dt=0.1          -> shared init, dynamics intact
# With P1's recurrent runs this completes the 2x2:
# {random, deterministic} x {frozen, dynamic}.
guard 5 "ablate_frozen_random" && \
python main.py -m $T $R model.dt=0.0 train.seed=0,1,2,3,4 \
  wandb.group=48h_ablate_frozen_random

guard 5 "ablate_dense_open" && \
python main.py -m $T $R model.dt=0.0 model.deterministic_phase=true \
  train.seed=0,1,2,3,4 wandb.group=48h_ablate_dense_open

guard 5 "ablate_det_dynamic" && \
python main.py -m $T $R model.deterministic_phase=true \
  train.seed=0,1,2,3,4 wandb.group=48h_ablate_det_dynamic

# ---------------------------------------------------------------- P3 (25)
# Train-time T sweep (T=6 covered by P1). Each run's t_variance callback
# also gives the *test*-time curve, so this yields the full
# train-T x test-T generalisation matrix.
guard 25 "train_T_sweep" && \
python main.py -m $T $R model.T=1,2,4,8,12 train.seed=0,1,2,3,4 \
  wandb.group=48h_train_T

# ---------------------------------------------------------------- P4 (12)
# Capacity: module count and module width.
guard 6 "n_modules" && \
python main.py -m $T $R model.n_modules=2,8 train.seed=0,1,2 \
  wandb.group=48h_n_modules

guard 6 "module_dim" && \
python main.py -m $T $R model.module_dim=64,256 train.seed=0,1,2 \
  wandb.group=48h_module_dim

# ---------------------------------------------------------------- P5 (9)
# Phase-dynamics timescale.
guard 9 "dt_sweep" && \
python main.py -m $T $R model.dt=0.05,0.2,0.5 train.seed=0,1,2 \
  wandb.group=48h_dt

# ---------------------------------------------------------------- P6 (6)
# Natural-frequency spread: omega=0 (no intrinsic drift) vs strong drift.
guard 6 "omega_sweep" && \
python main.py -m $T $R model.omega_init=0.0,2.0 train.seed=0,1,2 \
  wandb.group=48h_omega

echo "[queue] done at $(date)"
