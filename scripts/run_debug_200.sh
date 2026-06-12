#!/usr/bin/env bash
set -euo pipefail

cd /home/tahiti/ARTeccv
source .venv/bin/activate

python src/run_full_pipeline.py \
  --config configs/art_method.yaml \
  --max-images 200 \
  --resume
