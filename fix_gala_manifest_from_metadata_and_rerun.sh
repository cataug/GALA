#!/usr/bin/env bash
set -euo pipefail

cd /home/tahiti/ARTeccv

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

echo "============================================================"
echo "1) Build WikiArt manifest from metadata, not folders"
echo "============================================================"

cat > src/build_wikiart_manifest_from_metadata.py <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

BAD = {
    "", "nan", "none", "null", "unknown", "unk", "na", "n/a",
    "other", "misc", "undefined", "not available", "not_available"
}

PATH_CANDIDATES = [
    "image", "image_path", "path", "filepath", "file_path",
    "file_name", "filename", "file", "img_path", "rel_path"
]

LABEL_HINTS = [
    "artist", "style", "genre", "label", "class", "category",
    "movement", "school", "period", "tag", "target"
]


def clean(x):
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    if s.startswith("{") and "path" in s:
        try:
            d = json.loads(s.replace("'", '"'))
            s = d.get("path", s)
        except Exception:
            pass
    s = s.replace("\\", "/")
    s = re.sub(r"\.[a-zA-Z0-9]+$", "", s)
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return "" if s in BAD else s


def label_value(x, prefix="label"):
    s = clean(x)
    if s == "":
        return ""
    if re.fullmatch(r"-?\d+(\.0)?", s):
        s = str(int(float(s)))
        return f"{prefix}_{s}"
    return s


def find_root():
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
            n = sum(1 for x in p.rglob("*") if x.is_file() and x.suffix.lower() in IMG_EXTS)
            if n > 0:
                return p.resolve()
    raise FileNotFoundError("WikiArt root not found. Set WIKIART_ROOT=/path/to/wikiart")


def collect_images(root):
    imgs = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            rel = p.relative_to(root)
            imgs.append({
                "image_path": str(p),
                "rel_path": str(rel),
                "filename": p.name,
                "stem": p.stem,
                "basename": p.name,
            })
    return pd.DataFrame(imgs)


def metadata_files(root):
    pats = ["*.csv", "*.tsv", "*.jsonl", "*.json", "*.parquet"]
    files = []
    search_roots = [root, root.parent]
    for sr in search_roots:
        for pat in pats:
            files.extend(sr.rglob(pat))
    files = sorted(set(files), key=lambda p: (len(str(p)), str(p)))
    return files


def read_meta_file(p):
    try:
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p)
        if p.suffix.lower() == ".tsv":
            return pd.read_csv(p, sep="\t")
        if p.suffix.lower() == ".jsonl":
            rows = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return pd.DataFrame(rows)
        if p.suffix.lower() == ".json":
            obj = json.load(open(p, "r", encoding="utf-8"))
            if isinstance(obj, list):
                return pd.DataFrame(obj)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, list) and len(v) > 10:
                        return pd.DataFrame(v)
                return pd.DataFrame([obj])
        if p.suffix.lower() == ".parquet":
            return pd.read_parquet(p)
    except Exception as e:
        print(f"[skip metadata] {p}: {e}")
    return None


def path_from_cell(x):
    if isinstance(x, dict):
        return str(x.get("path", x.get("file_name", x.get("filename", ""))))
    s = str(x)
    if "path" in s and "{" in s:
        try:
            d = json.loads(s.replace("'", '"'))
            return str(d.get("path", ""))
        except Exception:
            return s
    return s


def find_path_col(df):
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for c in PATH_CANDIDATES:
        if c in cols:
            return c
        if c.lower() in low:
            return low[c.lower()]

    for c in cols:
        cl = c.lower()
        if any(x in cl for x in ["image", "file", "path"]):
            return c
    return None


def label_candidates(df):
    out = []
    for c in df.columns:
        cl = c.lower()
        if any(h in cl for h in LABEL_HINTS):
            out.append(c)

    # also include low-cardinality integer/string columns
    for c in df.columns:
        if c in out:
            continue
        try:
            nunique = df[c].nunique(dropna=True)
            if 2 <= nunique <= max(5000, len(df) // 2):
                if df[c].dtype != "object" or nunique < len(df) * 0.7:
                    out.append(c)
        except Exception:
            pass

    return out


def score_labels(df, cols):
    rows = []
    for c in cols:
        vals = df[c].map(lambda x: label_value(x, prefix=c))
        vals = vals[~vals.isin(BAD)]
        vc = vals.value_counts()
        vc2 = vc[vc >= 2]
        rows.append({
            "col": c,
            "n_valid": int(vc2.sum()),
            "n_classes": int(len(vc2)),
            "min_count": int(vc2.min()) if len(vc2) else 0,
            "median_count": float(vc2.median()) if len(vc2) else 0.0,
            "max_count": int(vc2.max()) if len(vc2) else 0,
        })
    if not rows:
        return pd.DataFrame(columns=["col", "n_valid", "n_classes"])
    return pd.DataFrame(rows).sort_values(["n_classes", "n_valid"], ascending=[False, False])


def join_meta_to_images(imgs, meta, root, meta_file):
    meta = meta.copy()
    pcol = find_path_col(meta)

    if pcol is not None:
        meta["__meta_path__"] = meta[pcol].map(path_from_cell)
        meta["__basename__"] = meta["__meta_path__"].map(lambda x: Path(str(x).replace("\\", "/")).name)
        meta["__stem__"] = meta["__basename__"].map(lambda x: Path(x).stem)

        imgs2 = imgs.copy()
        imgs2["__basename__"] = imgs2["basename"]
        imgs2["__stem__"] = imgs2["stem"]

        # basename join
        m1 = imgs2.merge(
            meta.drop_duplicates("__basename__"),
            on="__basename__",
            how="left",
            suffixes=("", "__meta")
        )
        hit = m1[pcol].notna().mean() if pcol in m1.columns else 0.0
        if hit > 0.2:
            print(f"[join] {meta_file.name}: by basename, hit={hit:.3f}")
            return m1

        # stem join
        m2 = imgs2.merge(
            meta.drop_duplicates("__stem__"),
            on="__stem__",
            how="left",
            suffixes=("", "__meta")
        )
        hit = m2[pcol].notna().mean() if pcol in m2.columns else 0.0
        if hit > 0.2:
            print(f"[join] {meta_file.name}: by stem, hit={hit:.3f}")
            return m2

    # positional fallback if same size
    if abs(len(meta) - len(imgs)) <= max(10, int(0.02 * len(imgs))):
        n = min(len(meta), len(imgs))
        print(f"[join] {meta_file.name}: positional, n={n}")
        return pd.concat(
            [imgs.iloc[:n].reset_index(drop=True), meta.iloc[:n].reset_index(drop=True)],
            axis=1
        )

    return None


def try_hf_dataset(root, imgs):
    try:
        from datasets import load_from_disk, load_dataset
    except Exception:
        return None

    # load_from_disk
    try:
        ds = load_from_disk(str(root))
        if hasattr(ds, "keys"):
            key = list(ds.keys())[0]
            ds = ds[key]
        df = ds.to_pandas()
        print(f"[hf] load_from_disk ok: rows={len(df)}, cols={list(df.columns)[:20]}")
        if len(df) > 0:
            tmp_path = Path("hf_load_from_disk")
            return join_meta_to_images(imgs, df, root, tmp_path) or pd.concat(
                [imgs.iloc[:min(len(imgs), len(df))].reset_index(drop=True),
                 df.iloc[:min(len(imgs), len(df))].reset_index(drop=True)],
                axis=1
            )
    except Exception as e:
        print(f"[hf skip] load_from_disk: {e}")

    # imagefolder
    try:
        ds = load_dataset("imagefolder", data_dir=str(root), split="train")
        df = ds.to_pandas()
        print(f"[hf] imagefolder ok: rows={len(df)}, cols={list(df.columns)[:20]}")
        if len(df) > 0:
            tmp_path = Path("hf_imagefolder")
            return join_meta_to_images(imgs, df, root, tmp_path) or pd.concat(
                [imgs.iloc[:min(len(imgs), len(df))].reset_index(drop=True),
                 df.iloc[:min(len(imgs), len(df))].reset_index(drop=True)],
                axis=1
            )
    except Exception as e:
        print(f"[hf skip] imagefolder: {e}")

    return None


def choose_best_metadata(root, imgs):
    best = None
    best_score = None
    best_diag = None
    best_file = None

    files = metadata_files(root)
    print(f"[metadata files found] {len(files)}")
    for f in files[:80]:
        print("  ", f)

    for f in files[:200]:
        meta = read_meta_file(f)
        if meta is None or len(meta) < 10:
            continue

        joined = join_meta_to_images(imgs, meta, root, f)
        if joined is None:
            continue

        cols = label_candidates(joined)
        diag = score_labels(joined, cols)

        if len(diag) == 0:
            continue

        top = diag.iloc[0]
        score = (int(top["n_classes"]), int(top["n_valid"]))

        print(f"\n[metadata candidate] {f}")
        print(diag.head(12).to_string(index=False))

        if best_score is None or score > best_score:
            best = joined
            best_score = score
            best_diag = diag
            best_file = f

    if best is not None and best_score[0] >= 2:
        print(f"\n[best metadata] {best_file}, score={best_score}")
        return best, best_diag, str(best_file)

    print("\n[metadata fallback] trying HuggingFace dataset readers")
    hf = try_hf_dataset(root, imgs)
    if hf is not None:
        cols = label_candidates(hf)
        diag = score_labels(hf, cols)
        print(diag.head(20).to_string(index=False))
        if len(diag) and int(diag.iloc[0]["n_classes"]) >= 2:
            return hf, diag, "hf_dataset"

    return None, None, None


def pick_main_columns(df, diag):
    usable = diag[(diag["n_classes"] >= 2) & (diag["n_valid"] >= 10)].copy()
    if len(usable) == 0:
        raise RuntimeError("Metadata exists, but no usable class column found.")

    cols = list(usable["col"])

    def pick_contains(words, fallback_idx=0):
        for c in cols:
            cl = c.lower()
            if any(w in cl for w in words):
                return c
        return cols[min(fallback_idx, len(cols)-1)]

    artist_col = pick_contains(["artist", "author", "painter"], 0)
    style_col = pick_contains(["style", "movement", "school"], 0)
    genre_col = pick_contains(["genre", "category"], 0)
    label_col = pick_contains(["artist", "style", "genre", "label", "class", "category"], 0)

    # Prefer many-class target for retrieval.
    label_col = usable.sort_values(["n_classes", "n_valid"], ascending=[False, False]).iloc[0]["col"]

    return str(label_col), str(artist_col), str(style_col), str(genre_col)


def balanced_sample(df, target_col, max_images=5000, min_per_class=2, max_per_class=80, seed=42):
    rng = np.random.default_rng(seed)

    tmp = df.copy()
    tmp["__target__"] = tmp[target_col].map(lambda x: label_value(x, prefix=target_col))
    tmp = tmp[~tmp["__target__"].isin(BAD)].copy()

    vc = tmp["__target__"].value_counts()
    good = vc[vc >= min_per_class]

    if len(good) < 2:
        raise RuntimeError(f"Target {target_col} has fewer than 2 classes.")

    # Limit number of classes so min_per_class does not exceed max_images.
    max_classes = max(2, max_images // min_per_class)
    good = good.head(max_classes)

    selected = []
    used = set()

    for cls in good.index:
        idxs = tmp.index[tmp["__target__"] == cls].to_numpy()
        rng.shuffle(idxs)
        take = idxs[:min_per_class]
        selected.extend(take.tolist())
        used.update(take.tolist())

    per_class = Counter(tmp.loc[selected, "__target__"])

    pool = tmp.index[tmp["__target__"].isin(good.index)].to_numpy()
    rng.shuffle(pool)

    for idx in pool:
        if len(selected) >= max_images:
            break
        if idx in used:
            continue
        cls = tmp.loc[idx, "__target__"]
        if per_class[cls] >= max_per_class:
            continue
        selected.append(int(idx))
        used.add(int(idx))
        per_class[cls] += 1

    out = tmp.loc[selected].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out = out.drop(columns=["__target__"], errors="ignore")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--out", default="outputs/wikiart_manifest_balanced.csv")
    ap.add_argument("--max_images", type=int, default=5000)
    ap.add_argument("--min_per_class", type=int, default=2)
    ap.add_argument("--max_per_class", type=int, default=80)
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else find_root()
    print("WIKIART_ROOT =", root)

    imgs = collect_images(root)
    print("Total images found:", len(imgs))
    print("\nExample image paths:")
    for x in imgs["rel_path"].head(20):
        print(" ", x)

    df, diag, source = choose_best_metadata(root, imgs)

    if df is None:
        raise RuntimeError(
            "No metadata with labels found. Need a CSV/JSONL/Parquet with artist/style/genre/label. "
            "Image paths alone are not enough because this WikiArt copy is flat."
        )

    print("\nBest metadata diagnostics:")
    print(diag.head(30).to_string(index=False))

    label_col, artist_col, style_col, genre_col = pick_main_columns(df, diag)

    print("\nChosen columns:")
    print("label :", label_col)
    print("artist:", artist_col)
    print("style :", style_col)
    print("genre :", genre_col)

    sampled = balanced_sample(
        df,
        target_col=label_col,
        max_images=args.max_images,
        min_per_class=args.min_per_class,
        max_per_class=args.max_per_class,
        seed=42,
    )

    sampled["label"] = sampled[label_col].map(lambda x: label_value(x, prefix=label_col))
    sampled["artist"] = sampled[artist_col].map(lambda x: label_value(x, prefix=artist_col))
    sampled["style"] = sampled[style_col].map(lambda x: label_value(x, prefix=style_col))
    sampled["genre"] = sampled[genre_col].map(lambda x: label_value(x, prefix=genre_col))

    sampled["sample_id"] = [f"wikiart_{i:06d}" for i in range(len(sampled))]
    sampled["image_id"] = sampled["sample_id"]
    sampled["idx"] = np.arange(len(sampled))

    keep_first = [
        "sample_id", "image_id", "idx",
        "image_path", "rel_path", "filename", "stem",
        "label", "artist", "style", "genre",
    ]
    cols = [c for c in keep_first if c in sampled.columns] + [c for c in sampled.columns if c not in keep_first]
    sampled = sampled[cols]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out, index=False)

    # Save aliases because old pipeline may expect these names.
    sampled.to_csv("outputs/wikiart_manifest_parent.csv", index=False)
    sampled.to_csv("outputs/wikiart_manifest.csv", index=False)

    rows = []
    for c in ["label", "artist", "style", "genre"]:
        vc = sampled[c].value_counts()
        rows.append({
            "label_col": c,
            "n_total": len(sampled),
            "n_valid": int(vc.sum()),
            "n_classes": int(len(vc)),
            "min_class_count": int(vc.min()),
            "median_class_count": float(vc.median()),
            "max_class_count": int(vc.max()),
        })
    diag2 = pd.DataFrame(rows)
    diag2.to_csv("outputs/wikiart_manifest_balanced_label_diagnostics.csv", index=False)

    print("\nFinal manifest diagnostics:")
    print(diag2.to_string(index=False))

    print("\nTop classes for label:")
    print(sampled["label"].value_counts().head(25).to_string())

    if sampled["label"].nunique() <= 1:
        raise RuntimeError("Still degenerate: label has one class.")

    print("\nSaved:")
    print(out)
    print("outputs/wikiart_manifest_parent.csv")
    print("outputs/wikiart_manifest.csv")
    print("outputs/wikiart_manifest_balanced_label_diagnostics.csv")
    print("\nOK: balanced manifest is ready.")


if __name__ == "__main__":
    main()
PY

python src/build_wikiart_manifest_from_metadata.py \
  --root /home/tahiti/UME2026/data/wikiart \
  --out outputs/wikiart_manifest_balanced.csv \
  --max_images 5000 \
  --min_per_class 2 \
  --max_per_class 80

echo ""
echo "============================================================"
echo "2) Patch config to balanced manifest"
echo "============================================================"

python - <<'PY'
from pathlib import Path
import re

cfg = Path("configs/art_method.yaml")
if cfg.exists():
    s = cfg.read_text()

    keys = ["manifest_path", "manifest_csv", "manifest"]
    changed = False
    for k in keys:
        pat = rf"(?m)^{k}\s*:\s*.*$"
        if re.search(pat, s):
            s = re.sub(pat, f"{k}: outputs/wikiart_manifest_balanced.csv", s)
            changed = True

    if not changed:
        s += "\nmanifest_path: outputs/wikiart_manifest_balanced.csv\n"

    s = re.sub(r"(?m)^max_images\s*:\s*.*$", "max_images: 5000", s)
    cfg.write_text(s)
    print("patched:", cfg)
else:
    print("configs/art_method.yaml not found, skip")
PY

echo ""
echo "============================================================"
echo "3) Clean current non-backup generated files"
echo "============================================================"

rm -rf outputs/generations outputs/embeddings outputs/fixed_eval
mkdir -p outputs/generations outputs/embeddings outputs/fixed_eval

echo ""
echo "============================================================"
echo "4) Run pipeline again"
echo "============================================================"

bash scripts/run_full_all.sh

echo ""
echo "============================================================"
echo "5) Recompute fixed retrieval metrics"
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
echo "6) Final check"
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

print(df[cols].head(80).to_string(index=False))

print("\nRandom:")
rnd = df[df["system"].astype(str).str.contains("random", case=False, na=False)]
print(rnd[cols].to_string(index=False))

print("\nBAD n_classes <= 1:", int((df["n_classes"] <= 1).sum()))

print("\nSaved files:")
print("outputs/wikiart_manifest_balanced.csv")
print("outputs/wikiart_manifest_balanced_label_diagnostics.csv")
print("outputs/fixed_eval/retrieval_metrics_fixed.csv")
print("outputs/fixed_eval/best_systems_by_target.csv")
print("outputs/fixed_eval/best_systems_by_target.tex")
print("outputs/summary_metrics_retrieval_fixed.csv")
PY
