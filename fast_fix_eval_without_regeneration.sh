#!/usr/bin/env bash
set -euo pipefail

cd /home/tahiti/ARTeccv

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

echo "============================================================"
echo "1) Restore latest backed-up generations/embeddings"
echo "============================================================"

GEN_BACKUP="$(ls -dt outputs/backups/generations_degenerate_* 2>/dev/null | head -1 || true)"
EMB_BACKUP="$(ls -dt outputs/backups/embeddings_degenerate_* 2>/dev/null | head -1 || true)"
MAN_BACKUP="$(ls -t outputs/backups/wikiart_manifest_parent.csv.degenerate_*.bak 2>/dev/null | head -1 || true)"

echo "GEN_BACKUP = $GEN_BACKUP"
echo "EMB_BACKUP = $EMB_BACKUP"
echo "MAN_BACKUP = $MAN_BACKUP"

if [ -z "$GEN_BACKUP" ]; then
  echo "ERROR: no backed-up generations found"
  exit 1
fi

rm -rf outputs/generations outputs/embeddings outputs/fixed_eval
mkdir -p outputs

cp -a "$GEN_BACKUP" outputs/generations

if [ -n "$EMB_BACKUP" ]; then
  cp -a "$EMB_BACKUP" outputs/embeddings
else
  mkdir -p outputs/embeddings
fi

if [ -n "$MAN_BACKUP" ]; then
  cp "$MAN_BACKUP" outputs/old_manifest_for_generated_rows.csv
fi

echo ""
echo "============================================================"
echo "2) Inspect what keys are inside old generations"
echo "============================================================"

python - <<'PY'
import json, glob
from pathlib import Path

files = sorted(glob.glob("outputs/generations/*.jsonl"))
print("generation files:", len(files))
for f in files[:5]:
    print("\nFILE:", f)
    with open(f, "r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                obj = json.loads(line)
                print("keys:", list(obj.keys()))
                for k, v in obj.items():
                    if isinstance(v, str):
                        print(k, "=", v[:160])
                    else:
                        print(k, "=", v)
                break
PY

echo ""
echo "============================================================"
echo "3) Build metadata labels only, no VLM regeneration"
echo "============================================================"

if [ ! -f src/build_wikiart_manifest_from_metadata.py ]; then
  echo "ERROR: src/build_wikiart_manifest_from_metadata.py not found."
  echo "Run the previous metadata-builder cell only until it creates this file."
  exit 1
fi

python src/build_wikiart_manifest_from_metadata.py \
  --root /home/tahiti/UME2026/data/wikiart \
  --out outputs/wikiart_manifest_balanced.csv \
  --max_images 5000 \
  --min_per_class 2 \
  --max_per_class 80

echo ""
echo "============================================================"
echo "4) Try fast evaluation using old generated outputs"
echo "============================================================"

python src/evaluate_fixed_retrieval.py \
  --manifest outputs/wikiart_manifest_balanced.csv \
  --generations_dir outputs/generations \
  --embeddings_dir outputs/embeddings \
  --out_dir outputs/fixed_eval \
  --min_class_count 2 \
  --ks 1 5 10 \
  --map_k 10 || true

echo ""
echo "============================================================"
echo "5) Final check"
echo "============================================================"

python - <<'PY'
from pathlib import Path
import pandas as pd

for p in [
    "outputs/wikiart_manifest_balanced_label_diagnostics.csv",
    "outputs/fixed_eval/label_diagnostics.csv",
    "outputs/fixed_eval/retrieval_metrics_fixed.csv",
    "outputs/summary_metrics_retrieval_fixed.csv",
]:
    print("\n===", p, "===")
    pp = Path(p)
    if pp.exists():
        try:
            df = pd.read_csv(pp)
            print(df.head(30).to_string(index=False))
        except Exception as e:
            print("exists but cannot preview:", e)
    else:
        print("missing")

p = Path("outputs/summary_metrics_retrieval_fixed.csv")
if p.exists():
    df = pd.read_csv(p)
    print("\nBAD n_classes <= 1:", int((df["n_classes"] <= 1).sum()) if "n_classes" in df else "no n_classes")
    rnd = df[df["system"].astype(str).str.contains("random", case=False, na=False)]
    if len(rnd):
        cols = [c for c in ["label_col","n_items","n_classes","recall@1","recall@5","recall@10","map@10"] if c in rnd.columns]
        print("\nRandom baseline:")
        print(rnd[cols].to_string(index=False))
PY

echo ""
echo "DONE. No VLM generation was run."
