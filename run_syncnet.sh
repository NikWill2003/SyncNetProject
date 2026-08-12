#!/usr/bin/env bash
set -uo pipefail

python prepare_dataset.py task=sort_of_clevr

python main.py experiment=sort_of_clevr/syncnet/conditioning

python main.py experiment=sort_of_clevr/syncnet/communication

python main.py experiment=sort_of_clevr/syncnet/dynamics

python main.py experiment=sort_of_clevr/syncnet/capacity