#!/usr/bin/env bash
set -euo pipefail

cd /home/tahiti/ARTeccv

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

echo "============================================================"
echo "1) Backup old degenerate outputs"
echo "============================================================"

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p outputs/backups

for d in outputs/generations outputs/embeddings outputs/fixed_eval; do
  if [ -d "$d" ]; then
    mv "$d" "outputs/backups/$(basename "$d")_degenerate_${TS}"
    echo "backup: $d -> outputs/backups/$(basename "$d")_degenerate_${TS}"
  fi
done

for f in outputs/summary_metrics.csv outputs/summary_metrics_retrieval_fixed.csv outputs/wikiart_manifest_parent.csv; do
  if [ -f "$f" ]; then
    cp "$f" "outputs/backups/$(basename "$f").degenerate_${TS}.bak"
    echo "backup file: $f"
  fi
done

mkdir -p outputs/generations outputs/embeddings outputs/fixed_eval

echo ""
echo "============================================================"
echo "2) Build balanced WikiArt manifest"
echo "============================================================"

cat > src/build_balanced_wikiart_manifest.py <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import random
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def clean_label(x):
    if x is None:
        return ""
    s = str(x).strip()
    s = s.replace("\\", "/")
    s = s.split("/")[-1]
    s = re.sub(r"\.[a-zA-Z0-9]+$", "", s)
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip().lower()
    bad = {"", "nan", "none", "null", "unknown", "other", "misc", "images", "image", "wikiart", "data"}
    return "" if s in bad else s


def artist_from_filename(path):
    stem = Path(path).stem

    # Common WikiArt filename format:
    # vincent-van-gogh_starry-night_1889.jpg
    if "_" in stem:
        first = stem.split("_")[0]
    else:
        # fallback, but avoid making every title into "artist"
        first = ""

    first = re.sub(r"\d+", "", first)
    first = clean_label(first)

    if len(first) < 3:
        return ""

    # Avoid labels that look like single generic words
    bad = {"image", "painting", "artwork", "wikiart"}
    if first in bad:
        return ""

    return first


def find_wikiart_root():
    candidates = [
        os.environ.get("WIKIART_ROOT", ""),
        "/home/tahiti/ARTeccv/data/wikiart",
        "/home/tahiti/UME2026/data/wikiart",
        "data/wikiart",
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if p.exists():
            imgs = list(p.rglob("*"))
            n = sum(1 for x in imgs if x.is_file() and x.suffix.lower() in IMG_EXTS)
            if n > 0:
                return p.resolve()
    raise FileNotFoundError("WikiArt root not found. Set WIKIART_ROOT=/path/to/wikiart")


def collect_images(root):
    rows = []

    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue

        rel = p.relative_to(root)
        parts = list(rel.parts[:-1])

        levels = {}
        for i in range(6):
            levels[f"level_{i}"] = clean_label(parts[i]) if i < len(parts) else ""

        parent = clean_label(parts[-1]) if len(parts) >= 1 else ""
        grandparent = clean_label(parts[-2]) if len(parts) >= 2 else ""
        greatgrandparent = clean_label(parts[-3]) if len(parts) >= 3 else ""

        fname_artist = artist_from_filename(p)

        rows.append({
            "image_path": str(p),
            "rel_path": str(rel),
            "filename": p.name,
            "stem": p.stem,
            "parent_label": parent,
            "grandparent_label": grandparent,
            "greatgrandparent_label": greatgrandparent,
            "artist_from_filename": fname_artist,
            **levels,
        })

    return pd.DataFrame(rows)


def usable_stats(df, col, min_count=2):
    s = df[col].fillna("").astype(str).map(clean_label)
    s = s[s != ""]
    vc = s.value_counts()
    vc = vc[vc >= min_count]
    return {
        "col": col,
        "n_valid": int(vc.sum()),
        "n_classes": int(len(vc)),
        "min_count": int(vc.min()) if len(vc) else 0,
        "median_count": float(vc.median()) if len(vc) else 0.0,
        "max_count": int(vc.max()) if len(vc) else 0,
    }


def choose_label_columns(df):
    candidates = [
        "artist_from_filename",
        "parent_label",
        "grandparent_label",
        "greatgrandparent_label",
        "level_0",
        "level_1",
        "level_2",
        "level_3",
        "level_4",
        "level_5",
    ]

    stats = []
    for c in candidates:
        if c in df.columns:
            st = usable_stats(df, c, min_count=2)
            stats.append(st)

    stats_df = pd.DataFrame(stats).sort_values(
        ["n_classes", "n_valid"],
        ascending=[False, False]
    )

    usable = stats_df[(stats_df["n_classes"] >= 2) & (stats_df["n_valid"] >= 100)]
    if len(usable) == 0:
        print(stats_df.to_string(index=False))
        raise RuntimeError("No usable labels found from WikiArt paths or filenames.")

    # Prefer artist from filename if it is usable.
    if "artist_from_filename" in set(usable["col"]):
        artist_col = "artist_from_filename"
    else:
        artist_col = usable.iloc[0]["col"]

    # Main target: many classes, enough repeated samples.
    label_col = usable.iloc[0]["col"]

    # Style/genre targets: path-based columns if available.
    path_usable = usable[usable["col"] != "artist_from_filename"]
    if len(path_usable) >= 1:
        style_col = path_usable.iloc[0]["col"]
    else:
        style_col = label_col

    if len(path_usable) >= 2:
        genre_col = path_usable.iloc[1]["col"]
    else:
        genre_col = style_col

    return stats_df, label_col, artist_col, style_col, genre_col


def balanced_sample(df, target_col, max_images=5000, min_per_class=2, max_per_class=80, seed=42):
    rng = random.Random(seed)

    df = df.copy()
    df[target_col] = df[target_col].fillna("").astype(str).map(clean_label)
    df = df[df[target_col] != ""].copy()

    groups = defaultdict(list)
    for idx, row in df.iterrows():
        groups[row[target_col]].append(idx)

    good = {k: v for k, v in groups.items() if len(v) >= min_per_class}
    if len(good) < 2:
        raise RuntimeError(f"Target {target_col} has fewer than 2 usable classes.")

    selected = []

    # First pass: guarantee minimum per class.
    classes = list(good.keys())
    rng.shuffle(classes)

    for c in classes:
        idxs = good[c][:]
        rng.shuffle(idxs)
        selected.extend(idxs[:min_per_class])

    # Second pass: fill up to max_images, capped per class.
    per_class = Counter(df.loc[selected, target_col])
    pool = []

    for c in classes:
        idxs = good[c][:]
        rng.shuffle(idxs)
        for idx in idxs:
            if idx not in selected:
                pool.append(idx)

    rng.shuffle(pool)

    for idx in pool:
        if len(selected) >= max_images:
            break
        c = df.loc[idx, target_col]
        if per_class[c] >= max_per_class:
            continue
        selected.append(idx)
        per_class[c] += 1

    out = df.loc[selected].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def main():
    root = find_wikiart_root()
    print(f"WIKIART_ROOT = {root}")

    df = collect_images(root)
    print(f"Total images found: {len(df)}")

    stats_df, label_col, artist_col, style_col, genre_col = choose_label_columns(df)

    print("\nCandidate label diagnostics:")
    print(stats_df.to_string(index=False))

    print("\nChosen columns:")
    print(f"label  = {label_col}")
    print(f"artist = {artist_col}")
    print(f"style  = {style_col}")
    print(f"genre  = {genre_col}")

    sampled = balanced_sample(
        df,
        target_col=label_col,
        max_images=5000,
        min_per_class=2,
        max_per_class=80,
        seed=42,
    )

    sampled["label"] = sampled[label_col].map(clean_label)
    sampled["artist"] = sampled[artist_col].map(clean_label)
    sampled["style"] = sampled[style_col].map(clean_label)
    sampled["genre"] = sampled[genre_col].map(clean_label)

    sampled["sample_id"] = [f"wikiart_{i:06d}" for i in range(len(sampled))]
    sampled["image_id"] = sampled["sample_id"]
    sampled["idx"] = range(len(sampled))

    # Keep columns expected by generation/evaluation scripts first.
    first_cols = [
        "sample_id",
        "image_id",
        "idx",
        "image_path",
        "rel_path",
        "filename",
        "label",
        "artist",
        "style",
        "genre",
        "parent_label",
        "grandparent_label",
        "greatgrandparent_label",
        "artist_from_filename",
        "level_0",
        "level_1",
        "level_2",
        "level_3",
        "level_4",
        "level_5",
    ]

    cols = [c for c in first_cols if c in sampled.columns] + [
        c for c in sampled.columns if c not in first_cols
    ]
    sampled = sampled[cols]

    out_path = Path("outputs/wikiart_manifest_balanced.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out_path, index=False)

    diag_path = Path("outputs/wikiart_manifest_balanced_label_diagnostics.csv")
    rows = []
    for c in ["label", "artist", "style", "genre", "parent_label", "grandparent_label", "artist_from_filename", "level_0", "level_1", "level_2"]:
        if c in sampled.columns:
            rows.append(usable_stats(sampled, c, min_count=2))
    pd.DataFrame(rows).to_csv(diag_path, index=False)

    print(f"\nSaved manifest: {out_path}")
    print(f"Saved diagnostics: {diag_path}")

    print("\nFinal manifest diagnostics:")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nTop label counts:")
    print(sampled["label"].value_counts().head(20).to_string())

    if sampled["label"].nunique() <= 1:
        raise RuntimeError("Balanced manifest is still degenerate. Check WikiArt folder structure.")

    print("\nOK: balanced manifest created.")


if __name__ == "__main__":
    main()
PY

python src/build_balanced_wikiart_manifest.py

echo ""
echo "============================================================"
echo "3) Patch config to use balanced manifest"
echo "============================================================"

python - <<'PY'
from pathlib import Path
import re

cfg = Path("configs/art_method.yaml")
if not cfg.exists():
    print("WARNING: configs/art_method.yaml not found, skipping config patch.")
    raise SystemExit(0)

s = cfg.read_text()

manifest_line_patterns = [
    r"(?m)^manifest_path\s*:\s*.*$",
    r"(?m)^manifest_csv\s*:\s*.*$",
    r"(?m)^manifest\s*:\s*.*$",
]

changed = False
for pat in manifest_line_patterns:
    if re.search(pat, s):
        s = re.sub(pat, "manifest_path: outputs/wikiart_manifest_balanced.csv", s)
        changed = True
        break

if not changed:
    s += "\nmanifest_path: outputs/wikiart_manifest_balanced.csv\n"

# Make sure max_images is 5000 if such key exists.
s = re.sub(r"(?m)^max_images\s*:\s*.*$", "max_images: 5000", s)

cfg.write_text(s)
print("patched:", cfg)
PY

echo ""
echo "============================================================"
echo "4) Patch evaluator so generation labels cannot override manifest labels"
echo "============================================================"

python - <<'PY'
from pathlib import Path

p = Path("src/evaluate_fixed_retrieval.py")
if not p.exists():
    print("WARNING: src/evaluate_fixed_retrieval.py not found. Recreate it from previous code first.")
    raise SystemExit(0)

s = p.read_text()

needle = '''def attach_manifest(df, manifest, source_name):
    df = df.copy()
'''
replacement = '''def attach_manifest(df, manifest, source_name):
    df = df.copy()

    # Drop labels stored inside generation files: they may come from an old
    # degenerate manifest. The external manifest is the source of truth.
    stale_label_cols = BASE_LABEL_COLS + [
        "parent_label", "grandparent_label", "greatgrandparent_label",
        "style_genre", "artist_style", "parent_grandparent"
    ]
    df = df.drop(columns=[c for c in stale_label_cols if c in df.columns], errors="ignore")
'''
if needle in s and "stale_label_cols = BASE_LABEL_COLS" not in s:
    s = s.replace(needle, replacement, 1)

start = s.find("    # Already has full labels")
end = s.find("    # 1) Join by manifest index-like columns")
if start != -1 and end != -1:
    s = s[:start] + "    # Always attach the external manifest.\n\n" + s[end:]

p.write_text(s)
print("patched:", p)
PY

echo ""
echo "============================================================"
echo "5) Run full pipeline again on balanced manifest"
echo "============================================================"

if [ -f scripts/run_full_all.sh ]; then
  bash scripts/run_full_all.sh
else
  echo "ERROR: scripts/run_full_all.sh not found."
  echo "Run your generation pipeline manually using outputs/wikiart_manifest_balanced.csv"
  exit 1
fi

echo ""
echo "============================================================"
echo "6) Recompute fixed retrieval metrics"
echo "============================================================"

python src/evaluate_fixed_retrieval.py \
  --manifest outputs/wikiart_manifest_balanced.csv \
  --generations_dir outputs/generations \
  --embeddings_dir outputs/embeddings \
  --out_dir outputs/fixed_eval \
  --min_class_count 2 \
  --ks 1 5 10 \
  --map_k 10

echo ""
echo "============================================================"
echo "7) Check final metrics"
echo "============================================================"

python - <<'PY'
import pandas as pd
from pathlib import Path

p = Path("outputs/summary_metrics_retrieval_fixed.csv")
if not p.exists():
    p = Path("outputs/fixed_eval/retrieval_metrics_fixed.csv")

df = pd.read_csv(p)

cols = ["label_col", "system", "system_type", "n_items", "n_classes", "recall@1", "recall@5", "recall@10", "map@10"]
cols = [c for c in cols if c in df.columns]

print("\n=== Metrics preview ===")
print(df[cols].head(60).to_string(index=False))

print("\n=== Random baseline ===")
rnd = df[df["system"].astype(str).str.contains("random", case=False, na=False)]
print(rnd[cols].to_string(index=False))

bad = df[df["n_classes"] <= 1]
print("\nRows with n_classes <= 1:", len(bad))

if len(rnd) and (rnd["recall@1"] >= 0.95).any():
    print("\nWARNING: random baseline is still too high for at least one target.")
    print("Use a more fine-grained label_col from outputs/fixed_eval/label_diagnostics.csv")
else:
    print("\nOK: retrieval is no longer degenerate.")

print("\nSaved:")
print("outputs/wikiart_manifest_balanced.csv")
print("outputs/fixed_eval/label_diagnostics.csv")
print("outputs/fixed_eval/retrieval_metrics_fixed.csv")
print("outputs/fixed_eval/best_systems_by_target.csv")
print("outputs/fixed_eval/best_systems_by_target.tex")
print("outputs/summary_metrics_retrieval_fixed.csv")
PY

