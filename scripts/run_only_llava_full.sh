#!/usr/bin/env bash
set -euo pipefail

cd /home/tahiti/ARTeccv
source .venv/bin/activate

python src/run_full_pipeline.py \
  --config configs/art_method.yaml \
  --models llava_onevision_0_5b \
  --ablations full \
  --resume
