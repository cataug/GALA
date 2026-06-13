#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import glob
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

BAD = {
    "", "nan", "none", "null", "unknown", "unk", "na", "n/a",
    "other", "misc", "undefined", "not available", "not_available"
}

PATH_KEYS = [
    "image_path", "path", "filepath", "file_path", "filename",
    "file", "image", "img_path", "local_path", "rel_path"
]


def clean(x, prefix="label"):
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    s = s.replace("\\", "/")
    s = re.sub(r"\.[a-zA-Z0-9]+$", "", s)
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    if s in BAD:
        return ""
    if re.fullmatch(r"-?\d+(\.0)?", s):
        return f"{prefix}_{int(float(s))}"
    return s


def read_first_generation_paths():
    files = sorted(glob.glob("outputs/generations/*.jsonl"))
    if not files:
        raise FileNotFoundError("No outputs/generations/*.jsonl found")

    rows = []
    f0 = files[0]
    with open(f0, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)

            path = ""
            for k in PATH_KEYS:
                if k in obj:
                    path = str(obj[k])
                    break

            rows.append({
                "idx": i,
                "sample_id": obj.get("sample_id", obj.get("image_id", f"gen_{i:06d}")),
                "image_id": obj.get("image_id", obj.get("sample_id", f"gen_{i:06d}")),
                "gen_path": path,
                "gen_basename": Path(path.replace("\\", "/")).name if path else "",
                "gen_stem": Path(path.replace("\\", "/")).stem if path else "",
            })

    df = pd.DataFrame(rows)
    print("Using generation file:", f0)
    print("Generation rows:", len(df))
    print("Generation columns preview:")
    print(df.head(5).to_string(index=False))
    return df


def collect_wikiart_images(root):
    rows = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            rows.append({
                "image_path": str(p),
                "rel_path": str(p.relative_to(root)),
                "filename": p.name,
                "basename": p.name,
                "stem": p.stem,
            })
    return pd.DataFrame(rows)


def read_meta_files(root):
    files = []
    for pat in ["*.csv", "*.tsv", "*.jsonl", "*.json", "*.parquet"]:
        files += list(Path(root).rglob(pat))
        files += list(Path(root).parent.rglob(pat))

    files = sorted(set(files), key=lambda x: (len(str(x)), str(x)))
    out = []

    for p in files:
        try:
            if p.suffix == ".csv":
                df = pd.read_csv(p)
            elif p.suffix == ".tsv":
                df = pd.read_csv(p, sep="\t")
            elif p.suffix == ".jsonl":
                data = []
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                df = pd.DataFrame(data)
            elif p.suffix == ".json":
                obj = json.load(open(p, "r", encoding="utf-8"))
                if isinstance(obj, list):
                    df = pd.DataFrame(obj)
                elif isinstance(obj, dict):
                    best = None
                    for v in obj.values():
                        if isinstance(v, list) and len(v) > 10:
                            best = v
                            break
                    df = pd.DataFrame(best if best is not None else [obj])
                else:
                    continue
            elif p.suffix == ".parquet":
                df = pd.read_parquet(p)
            else:
                continue

            if len(df) >= 10:
                out.append((p, df))
        except Exception:
            pass

    return out


def find_path_col(df):
    for c in df.columns:
        cl = c.lower()
        if any(x in cl for x in ["image", "file", "path", "filename"]):
            return c
    return None


def label_cols(df):
    cols = []
    for c in df.columns:
        cl = c.lower()
        if any(x in cl for x in ["artist", "style", "genre", "label", "class", "category", "movement", "school"]):
            cols.append(c)
    return cols


def score_join(joined):
    labs = label_cols(joined)
    best = []
    for c in labs:
        vals = joined[c].map(lambda x: clean(x, prefix=c))
        vc = vals[vals != ""].value_counts()
        vc = vc[vc >= 2]
        best.append((c, int(len(vc)), int(vc.sum())))
    best = sorted(best, key=lambda x: (x[1], x[2]), reverse=True)
    return best


def main():
    root = Path("/home/tahiti/UME2026/data/wikiart")
    gen = read_first_generation_paths()

    imgs = collect_wikiart_images(root)
    print("\nWikiArt images:", len(imgs))

    metas = read_meta_files(root)
    print("Metadata files:", len(metas))

    best_join = None
    best_score = (-1, -1)
    best_info = None

    # First create image table aligned to generation paths.
    gen2 = gen.copy()

    if gen2["gen_basename"].astype(str).str.len().sum() > 0:
        img_join = gen2.merge(
            imgs.drop_duplicates("basename"),
            left_on="gen_basename",
            right_on="basename",
            how="left"
        )
        hit = img_join["image_path"].notna().mean()
        print("\nImage join by basename hit:", round(hit, 4))
    else:
        img_join = None
        hit = 0

    if img_join is None or hit < 0.5:
        print("WARNING: generation files do not expose usable image paths.")
        print("Then old generations cannot be safely relabeled; full regeneration is needed.")
        return

    for mf, meta in metas:
        pcol = find_path_col(meta)
        if pcol is None:
            continue

        meta = meta.copy()
        meta["meta_path"] = meta[pcol].astype(str)
        meta["meta_basename"] = meta["meta_path"].map(lambda x: Path(x.replace("\\", "/")).name)
        meta["meta_stem"] = meta["meta_path"].map(lambda x: Path(x.replace("\\", "/")).stem)

        joined = img_join.merge(
            meta.drop_duplicates("meta_basename"),
            left_on="basename",
            right_on="meta_basename",
            how="left",
            suffixes=("", "__meta")
        )

        hit_meta = joined[pcol].notna().mean() if pcol in joined.columns else 0.0
        scores = score_join(joined)

        if scores:
            top_classes, top_valid = scores[0][1], scores[0][2]
            score = (top_classes, top_valid)
            print("\nMETA:", mf)
            print("hit:", round(hit_meta, 4), "top:", scores[:8])

            if hit_meta > 0.5 and score > best_score:
                best_join = joined
                best_score = score
                best_info = (mf, pcol, scores)

    if best_join is None:
        print("\nERROR: no metadata could be joined to generated images.")
        print("Full regeneration with metadata manifest is needed.")
        return

    mf, pcol, scores = best_info
    print("\nBEST META:", mf)
    print("path col:", pcol)
    print("scores:", scores[:12])

    # Pick columns.
    available = [x[0] for x in scores]

    def pick(names, fallback):
        for c in available:
            cl = c.lower()
            if any(n in cl for n in names):
                return c
        return fallback

    main_label = available[0]
    artist_col = pick(["artist", "author", "painter"], main_label)
    style_col = pick(["style", "movement", "school"], main_label)
    genre_col = pick(["genre", "category"], main_label)

    out = best_join.copy()
    out["label"] = out[main_label].map(lambda x: clean(x, prefix=main_label))
    out["artist"] = out[artist_col].map(lambda x: clean(x, prefix=artist_col))
    out["style"] = out[style_col].map(lambda x: clean(x, prefix=style_col))
    out["genre"] = out[genre_col].map(lambda x: clean(x, prefix=genre_col))

    # Drop rows without main labels.
    out = out[out["label"] != ""].copy()
    out = out.reset_index(drop=True)

    out["sample_id"] = [f"existing_{i:06d}" for i in range(len(out))]
    out["image_id"] = out["sample_id"]
    out["idx"] = range(len(out))

    out["style_genre"] = out["style"] + "__" + out["genre"]
    out["artist_style"] = out["artist"] + "__" + out["style"]

    keep = [
        "sample_id", "image_id", "idx",
        "image_path", "rel_path", "filename",
        "label", "artist", "style", "genre", "artist_style", "style_genre",
        "gen_path", "gen_basename", "gen_stem"
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep + [c for c in out.columns if c not in keep]]

    out_path = Path("outputs/manifest_for_existing_generations.csv")
    out.to_csv(out_path, index=False)

    diag = []
    for c in ["label", "artist", "style", "genre", "artist_style", "style_genre"]:
        vc = out[c].value_counts()
        vc = vc[vc >= 2]
        diag.append({
            "label_col": c,
            "n_total": len(out),
            "n_classes": len(vc),
            "n_valid": int(vc.sum()),
            "min_class_count": int(vc.min()) if len(vc) else 0,
            "median_class_count": float(vc.median()) if len(vc) else 0.0,
            "max_class_count": int(vc.max()) if len(vc) else 0,
        })

    diag = pd.DataFrame(diag)
    diag.to_csv("outputs/manifest_for_existing_generations_diagnostics.csv", index=False)

    print("\nSaved:", out_path)
    print(diag.to_string(index=False))

    if diag["n_classes"].max() <= 1:
        raise RuntimeError("Still degenerate labels for existing generations.")


if __name__ == "__main__":
    main()
