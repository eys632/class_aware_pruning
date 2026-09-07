#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
python experiment.py \
  --seeds 0 1 2 3 4 \
  --epochs 25 \
  --batch-size 2048 \
  --prune-ratios 0.1 0.2 0.3 0.4 0.5 \
  --out-dir runs/global_5seeds
