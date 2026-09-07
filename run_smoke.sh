#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
python experiment.py \
  --seeds 0 \
  --epochs 3 \
  --prune-ratios 0.2 0.5 \
  --out-dir runs/smoke
