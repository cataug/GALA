#!/usr/bin/env bash
set -euo pipefail

cd /home/tahiti/ARTeccv

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

python src/evaluate_fixed_retrieval.py \
  --manifest auto \
  --generations_dir outputs/generations \
  --embeddings_dir outputs/embeddings \
  --out_dir outputs/fixed_eval \
  --min_class_count 2 \
  --ks 1 5 10 \
  --map_k 10
