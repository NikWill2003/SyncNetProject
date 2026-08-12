#!/usr/bin/env bash
set -uo pipefail

python prepare_dataset.py task=sort_of_clevr

# 10 runs -- the floor every other table is a delta above, so it runs first
python main.py experiment=sort_of_clevr/baselines/question_only

# 30 runs -- q_conditioning x patch_size {5, 15}
python main.py experiment=sort_of_clevr/baselines/transformer_conditioning

# 18 runs -- is the ceiling capacity or conditioning?
python main.py experiment=sort_of_clevr/baselines/transformer_capacity

# 18 runs -- looped vs stacked at matched depth
python main.py experiment=sort_of_clevr/baselines/transformer_depth