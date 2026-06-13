#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import math
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# Utils
# ============================================================

BAD_LABELS = {
    "", "nan", "none", "null", "unknown", "unk", "na", "n/a",
    "other", "misc", "undefined", "not_available", "not available"
}

ID_COLS = [
    "image_id", "sample_id", "uid", "id", "idx", "index",
    "row_id", "manifest_id", "manifest_index"
]

PATH_COLS = [
    "image_path", "path", "filepath", "file_path", "filename",
    "file", "img_path", "image", "local_path"
]

TEXT_COLS = [
    "response", "generated_text", "generation", "text", "caption",
    "description", "answer", "output"
]

BASE_LABEL_COLS = [
    "artist", "style", "genre", "period", "school", "label", "class",
    "category", "parent", "parent_label", "folder", "target"
]


def norm_str(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def norm_label(x):
    s = norm_str(x)
    s = s.replace("\\", "/")
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def safe_name(s):
    s = str(s)
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    return s.strip("_")


def find_first_existing(cols, candidates):
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def find_manifest(path_arg):
    if path_arg and path_arg != "auto":
        p = Path(path_arg)
        if not p.exists():
            raise FileNotFoundError(f"Manifest not found: {p}")
        return p

    candidates = []
    patterns = [
        "outputs/*manifest*.csv",
        "outputs/**/*manifest*.csv",
        "data/*manifest*.csv",
        "data/**/*manifest*.csv",
        "*manifest*.csv",
        "**/*manifest*.csv",
    ]
    for pat in patterns:
        candidates.extend(glob.glob(pat, recursive=True))

    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(
            "No manifest CSV found. Expected something like outputs/wikiart_manifest_parent.csv"
        )

    # Prefer output manifest, then shorter paths
    candidates = sorted(
        candidates,
        key=lambda x: (
            0 if str(x).startswith("outputs/") else 1,
            len(str(x)),
            str(x),
        ),
    )
    return Path(candidates[0])


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj["__jsonl_line__"] = line_no
                rows.append(obj)
            except Exception:
                continue
    return pd.DataFrame(rows)


def get_text_column(df):
    col = find_first_existing(df.columns, TEXT_COLS)
    if col is not None:
        return col

    # Fallback: choose longest object column
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not obj_cols:
        return None

    best = None
    best_len = -1
    for c in obj_cols:
        avg_len = df[c].fillna("").astype(str).str.len().mean()
        if avg_len > best_len:
            best_len = avg_len
            best = c
    return best


def extract_path_parts(path):
    s = norm_str(path).replace("\\", "/")
    if not s:
        return "", "", "", "", ""
    p = Path(s)
    parts = list(p.parts)
    basename = p.name
    stem = p.stem
    parent = parts[-2] if len(parts) >= 2 else ""
    grandparent = parts[-3] if len(parts) >= 3 else ""
    greatgrandparent = parts[-4] if len(parts) >= 4 else ""
    return basename, stem, parent, grandparent, greatgrandparent


# ============================================================
# Manifest preparation
# ============================================================

def prepare_manifest(manifest_path):
    df = pd.read_csv(manifest_path)
    df = df.copy()

    df["__manifest_row__"] = np.arange(len(df)).astype(int)
    df["__pos_id__"] = df["__manifest_row__"].astype(str)

    path_col = find_first_existing(df.columns, PATH_COLS)
    if path_col is not None:
        parts = df[path_col].apply(extract_path_parts)
        df["__basename__"] = parts.apply(lambda x: x[0])
        df["__stem__"] = parts.apply(lambda x: x[1])
        df["parent_label"] = parts.apply(lambda x: norm_label(x[2]))
        df["grandparent_label"] = parts.apply(lambda x: norm_label(x[3]))
        df["greatgrandparent_label"] = parts.apply(lambda x: norm_label(x[4]))
    else:
        df["__basename__"] = ""
        df["__stem__"] = ""
        df["parent_label"] = ""
        df["grandparent_label"] = ""
        df["greatgrandparent_label"] = ""

    # Normalize all existing likely labels
    for c in list(df.columns):
        if c.lower() in [x.lower() for x in BASE_LABEL_COLS]:
            df[c] = df[c].apply(norm_label)

    # Add combined labels when possible
    def combine(a, b):
        aa = df[a].apply(norm_label) if a in df.columns else pd.Series([""] * len(df))
        bb = df[b].apply(norm_label) if b in df.columns else pd.Series([""] * len(df))
        out = []
        for x, y in zip(aa, bb):
            if x and y and x not in BAD_LABELS and y not in BAD_LABELS:
                out.append(f"{x}__{y}")
            else:
                out.append("")
        return out

    if "style" in df.columns and "genre" in df.columns:
        df["style_genre"] = combine("style", "genre")
    if "artist" in df.columns and "style" in df.columns:
        df["artist_style"] = combine("artist", "style")
    if "parent_label" in df.columns and "grandparent_label" in df.columns:
        df["parent_grandparent"] = combine("grandparent_label", "parent_label")

    return df, path_col


def label_diagnostics(manifest, min_class_count):
    candidate_cols = []

    for c in manifest.columns:
        cl = c.lower()
        if (
            cl in [x.lower() for x in BASE_LABEL_COLS]
            or cl in {
                "parent_label", "grandparent_label", "greatgrandparent_label",
                "style_genre", "artist_style", "parent_grandparent"
            }
        ):
            candidate_cols.append(c)

    candidate_cols = sorted(set(candidate_cols))

    rows = []
    usable_cols = []
    for c in candidate_cols:
        labels = manifest[c].apply(norm_label)
        valid = ~labels.isin(BAD_LABELS)
        labels_valid = labels[valid]
        counts = labels_valid.value_counts()
        counts2 = counts[counts >= min_class_count]
        n_classes = int(len(counts2))
        n_valid = int(counts2.sum()) if n_classes > 0 else 0

        row = {
            "label_col": c,
            "n_total": len(labels),
            "n_valid_after_min_count": n_valid,
            "n_classes_after_min_count": n_classes,
            "min_class_count": int(counts2.min()) if n_classes else 0,
            "median_class_count": float(counts2.median()) if n_classes else 0.0,
            "max_class_count": int(counts2.max()) if n_classes else 0,
            "usable": bool(n_classes >= 2 and n_valid >= 3),
        }
        rows.append(row)
        if row["usable"]:
            usable_cols.append(c)

    diag = pd.DataFrame(rows).sort_values(
        ["usable", "n_classes_after_min_count", "n_valid_after_min_count"],
        ascending=[False, False, False]
    )

    return diag, usable_cols


# ============================================================
# Joining rows to manifest
# ============================================================

def attach_manifest(df, manifest, source_name):
    df = df.copy()

    # Already has full labels
    has_any_label = any(c in df.columns for c in BASE_LABEL_COLS + ["parent_label", "grandparent_label"])
    if has_any_label:
        for c in BASE_LABEL_COLS + ["parent_label", "grandparent_label", "greatgrandparent_label"]:
            if c in df.columns:
                df[c] = df[c].apply(norm_label)
        return df

    # 1) Join by manifest index-like columns
    for g_col in ID_COLS + ["__jsonl_line__"]:
        if g_col not in df.columns:
            continue
        for m_col in ["__manifest_row__"] + ID_COLS:
            if m_col not in manifest.columns:
                continue
            try:
                left = df[g_col].astype(str)
                right = manifest[m_col].astype(str)
                overlap = left.isin(set(right)).mean()
                if overlap > 0.5:
                    tmp = df.copy()
                    tmp["__join_key__"] = left
                    man = manifest.copy()
                    man["__join_key__"] = right
                    merged = tmp.merge(
                        man,
                        on="__join_key__",
                        how="left",
                        suffixes=("", "__manifest")
                    )
                    hit_rate = merged["__manifest_row__"].notna().mean()
                    if hit_rate > 0.5:
                        print(f"[join] {source_name}: by {g_col} -> {m_col}, hit={hit_rate:.3f}")
                        return merged
            except Exception:
                pass

    # 2) Join by path basename
    g_path_col = find_first_existing(df.columns, PATH_COLS)
    if g_path_col is not None and "__basename__" in manifest.columns:
        tmp = df.copy()
        tmp["__basename__"] = tmp[g_path_col].apply(lambda x: extract_path_parts(x)[0])
        man = manifest.drop_duplicates("__basename__").copy()
        merged = tmp.merge(
            man,
            on="__basename__",
            how="left",
            suffixes=("", "__manifest")
        )
        hit_rate = merged["__manifest_row__"].notna().mean()
        if hit_rate > 0.5:
            print(f"[join] {source_name}: by basename, hit={hit_rate:.3f}")
            return merged

    # 3) Positional fallback
    n = min(len(df), len(manifest))
    tmp = df.iloc[:n].reset_index(drop=True).copy()
    man = manifest.iloc[:n].reset_index(drop=True).copy()
    merged = pd.concat([tmp, man], axis=1)
    print(f"[join] {source_name}: positional fallback, rows={n}")
    return merged


# ============================================================
# Embeddings
# ============================================================

def tfidf_embeddings(texts):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            max_features=50000,
            norm="l2"
        )
        X = vec.fit_transform(texts)
        return X
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for TF-IDF text embeddings. "
            "Install with: pip install scikit-learn"
        ) from e


def load_npz_embedding(path):
    z = np.load(path, allow_pickle=True)
    files = list(z.files)

    X = None
    X_key = None
    preferred = [
        "embeddings", "embedding", "vectors", "X", "x",
        "image_embeds", "text_embeds", "features", "feats"
    ]

    for k in preferred + files:
        if k not in files:
            continue
        arr = z[k]
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
            X = arr.astype(np.float32)
            X_key = k
            break

    if X is None:
        return None, None, None

    ids = None
    id_key = None
    for k in ["ids", "image_ids", "sample_ids", "paths", "filenames", "keys", "index"]:
        if k in files:
            arr = z[k]
            if len(arr) == X.shape[0]:
                ids = arr.astype(str)
                id_key = k
                break

    return X, ids, X_key


# ============================================================
# Retrieval metrics
# ============================================================

def cosine_sim_matrix(X):
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(X)
    except Exception:
        X = np.asarray(X, dtype=np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        X = X / norms
        return X @ X.T


def average_precision_at_k(ranked_relevant, k, total_relevant):
    if total_relevant <= 0:
        return np.nan

    ranked_relevant = ranked_relevant[:k]
    hits = 0
    precisions = []
    for r, is_rel in enumerate(ranked_relevant, start=1):
        if is_rel:
            hits += 1
            precisions.append(hits / r)

    denom = min(total_relevant, k)
    if denom <= 0:
        return np.nan
    return float(np.sum(precisions) / denom)


def eval_from_similarity(S, labels, ks=(1, 5, 10), map_k=10):
    labels = np.asarray(labels)
    n = len(labels)
    assert S.shape[0] == S.shape[1] == n

    S = np.asarray(S, dtype=np.float32).copy()
    np.fill_diagonal(S, -np.inf)

    recalls = {k: [] for k in ks}
    aps = []
    valid_queries = 0

    for i in range(n):
        relevant = (labels == labels[i])
        relevant[i] = False
        total_rel = int(relevant.sum())

        if total_rel <= 0:
            continue

        valid_queries += 1
        order = np.argsort(-S[i])
        ranked_rel = relevant[order]

        for k in ks:
            kk = min(k, n - 1)
            recalls[k].append(float(ranked_rel[:kk].any()))

        aps.append(average_precision_at_k(ranked_rel, map_k, total_rel))

    out = {
        "n_items": int(n),
        "n_eval_queries": int(valid_queries),
        "n_classes": int(len(pd.Series(labels).unique())),
        "map@{}".format(map_k): float(np.nanmean(aps)) if aps else np.nan,
    }

    for k in ks:
        vals = recalls[k]
        out[f"recall@{k}"] = float(np.mean(vals)) if vals else np.nan

    return out


def eval_embeddings(X, labels, ks=(1, 5, 10), map_k=10):
    S = cosine_sim_matrix(X)
    return eval_from_similarity(S, labels, ks=ks, map_k=map_k)


def eval_random(labels, ks=(1, 5, 10), map_k=10, seeds=(13, 42, 777)):
    labels = np.asarray(labels)
    n = len(labels)

    all_rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)

        recalls = {k: [] for k in ks}
        aps = []
        valid_queries = 0

        for i in range(n):
            relevant = (labels == labels[i])
            relevant[i] = False
            total_rel = int(relevant.sum())
            if total_rel <= 0:
                continue

            valid_queries += 1
            candidates = np.array([j for j in range(n) if j != i])
            rng.shuffle(candidates)
            ranked_rel = relevant[candidates]

            for k in ks:
                kk = min(k, n - 1)
                recalls[k].append(float(ranked_rel[:kk].any()))

            aps.append(average_precision_at_k(ranked_rel, map_k, total_rel))

        row = {
            "n_items": int(n),
            "n_eval_queries": int(valid_queries),
            "n_classes": int(len(pd.Series(labels).unique())),
            "map@{}".format(map_k): float(np.nanmean(aps)) if aps else np.nan,
        }
        for k in ks:
            vals = recalls[k]
            row[f"recall@{k}"] = float(np.mean(vals)) if vals else np.nan
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    return df.mean(numeric_only=True).to_dict()


def filter_by_label(df, label_col, min_class_count):
    labels = df[label_col].apply(norm_label)
    valid = ~labels.isin(BAD_LABELS)
    counts = labels[valid].value_counts()
    good_classes = set(counts[counts >= min_class_count].index)

    mask = labels.isin(good_classes)
    idx = np.where(mask.values)[0]
    y = labels.iloc[idx].values

    if len(set(y)) < 2:
        return None, None

    return idx, y


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="auto")
    ap.add_argument("--generations_dir", default="outputs/generations")
    ap.add_argument("--embeddings_dir", default="outputs/embeddings")
    ap.add_argument("--out_dir", default="outputs/fixed_eval")
    ap.add_argument("--min_class_count", type=int, default=2)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--map_k", type=int, default=10)
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = find_manifest(args.manifest)
    print(f"[manifest] {manifest_path}")

    manifest, manifest_path_col = prepare_manifest(manifest_path)
    if args.max_rows and args.max_rows > 0:
        manifest = manifest.iloc[:args.max_rows].copy()

    diag, usable_label_cols = label_diagnostics(manifest, args.min_class_count)
    diag.to_csv(out_dir / "label_diagnostics.csv", index=False)

    print("\n[label diagnostics]")
    print(diag.head(20).to_string(index=False))

    if not usable_label_cols:
        raise RuntimeError(
            "No usable label columns found. Check outputs/fixed_eval/label_diagnostics.csv"
        )

    print("\n[usable label columns]")
    print(", ".join(usable_label_cols))

    results = []

    # --------------------------------------------------------
    # Text retrieval from generation JSONL files
    # --------------------------------------------------------
    gen_files = sorted(glob.glob(str(Path(args.generations_dir) / "*.jsonl")))
    print(f"\n[generation files] {len(gen_files)}")

    for gf in gen_files:
        gf = Path(gf)
        system = gf.stem
        gen = read_jsonl(gf)

        if len(gen) == 0:
            print(f"[skip] empty jsonl: {gf.name}")
            continue

        if args.max_rows and args.max_rows > 0:
            gen = gen.iloc[:args.max_rows].copy()

        text_col = get_text_column(gen)
        if text_col is None:
            print(f"[skip] no text column: {gf.name}")
            continue

        merged = attach_manifest(gen, manifest, system)
        texts = merged[text_col].fillna("").astype(str).tolist()

        # Drop empty texts only during embedding, but preserve row alignment
        print(f"[tfidf] {system}: rows={len(texts)}, text_col={text_col}")
        X = tfidf_embeddings(texts)

        for label_col in usable_label_cols:
            if label_col not in merged.columns:
                continue

            idx, y = filter_by_label(merged, label_col, args.min_class_count)
            if idx is None:
                continue

            X_sub = X[idx]
            metrics = eval_embeddings(X_sub, y, ks=args.ks, map_k=args.map_k)
            metrics.update({
                "system": system,
                "system_type": "vlm_text_tfidf",
                "label_col": label_col,
                "source_file": str(gf),
                "embedding_source": "tfidf_from_generation",
            })
            results.append(metrics)

    # --------------------------------------------------------
    # Existing NPZ embeddings
    # --------------------------------------------------------
    emb_files = sorted(glob.glob(str(Path(args.embeddings_dir) / "*.npz")))
    print(f"\n[embedding files] {len(emb_files)}")

    for ef in emb_files:
        ef = Path(ef)
        system = ef.stem

        if "random" in system.lower():
            continue

        X, ids, X_key = load_npz_embedding(ef)
        if X is None:
            print(f"[skip] cannot find 2D embedding array in {ef.name}")
            continue

        if args.max_rows and args.max_rows > 0:
            X = X[:args.max_rows]
            if ids is not None:
                ids = ids[:args.max_rows]

        # Attach manifest by ids or position
        if ids is not None:
            tmp = pd.DataFrame({"__emb_id__": ids})
            attached = attach_manifest(tmp, manifest, system)
        else:
            n = min(X.shape[0], len(manifest))
            X = X[:n]
            attached = manifest.iloc[:n].reset_index(drop=True).copy()

        if len(attached) != X.shape[0]:
            n = min(len(attached), X.shape[0])
            attached = attached.iloc[:n].reset_index(drop=True)
            X = X[:n]

        print(f"[npz] {system}: rows={X.shape[0]}, dim={X.shape[1]}, key={X_key}")

        for label_col in usable_label_cols:
            if label_col not in attached.columns:
                continue

            idx, y = filter_by_label(attached, label_col, args.min_class_count)
            if idx is None:
                continue

            X_sub = X[idx]
            metrics = eval_embeddings(X_sub, y, ks=args.ks, map_k=args.map_k)
            metrics.update({
                "system": system,
                "system_type": "npz_embedding",
                "label_col": label_col,
                "source_file": str(ef),
                "embedding_source": X_key,
            })
            results.append(metrics)

    # --------------------------------------------------------
    # Correct random baseline for each usable target
    # --------------------------------------------------------
    print("\n[random baselines]")
    for label_col in usable_label_cols:
        idx, y = filter_by_label(manifest, label_col, args.min_class_count)
        if idx is None:
            continue

        metrics = eval_random(y, ks=args.ks, map_k=args.map_k)
        metrics.update({
            "system": "baseline_random_fixed",
            "system_type": "random",
            "label_col": label_col,
            "source_file": "manifest",
            "embedding_source": "random_permutation_excluding_self",
        })
        results.append(metrics)

    if not results:
        raise RuntimeError("No metrics computed.")

    res = pd.DataFrame(results)

    # Sort columns
    metric_cols = [c for c in res.columns if c.startswith("recall@") or c.startswith("map@")]
    first_cols = [
        "label_col", "system", "system_type", "n_items", "n_eval_queries",
        "n_classes"
    ]
    other_cols = [c for c in res.columns if c not in first_cols + metric_cols]
    res = res[first_cols + metric_cols + other_cols]

    res_path = out_dir / "retrieval_metrics_fixed.csv"
    res.to_csv(res_path, index=False)

    # Also save to common root for paper scripts
    root_res_path = Path("outputs") / "summary_metrics_retrieval_fixed.csv"
    res.to_csv(root_res_path, index=False)

    # Best systems per target, excluding random
    sort_metric = "recall@1" if "recall@1" in res.columns else metric_cols[0]
    non_random = res[res["system_type"] != "random"].copy()
    best = (
        non_random.sort_values(["label_col", sort_metric], ascending=[True, False])
        .groupby("label_col")
        .head(8)
        .reset_index(drop=True)
    )

    best_path = out_dir / "best_systems_by_target.csv"
    best.to_csv(best_path, index=False)

    # Compact LaTeX table
    tex_cols = ["label_col", "system", "n_items", "n_classes"] + metric_cols
    tex_df = best[tex_cols].copy()
    tex_df.columns = [
        c.replace("_", "\\_").replace("@", "@") for c in tex_df.columns
    ]

    tex = tex_df.to_latex(
        index=False,
        escape=False,
        float_format=lambda x: f"{x:.3f}"
    )

    tex_path = out_dir / "best_systems_by_target.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)

    print("\n[done]")
    print(f"Saved: {res_path}")
    print(f"Saved: {root_res_path}")
    print(f"Saved: {best_path}")
    print(f"Saved: {tex_path}")
    print(f"Saved: {out_dir / 'label_diagnostics.csv'}")

    print("\n[top results]")
    show_cols = ["label_col", "system", "system_type", "n_items", "n_classes"] + metric_cols
    print(best[show_cols].head(30).to_string(index=False))

    print("\n[check random]")
    rnd = res[res["system"] == "baseline_random_fixed"]
    print(rnd[["label_col", "n_items", "n_classes"] + metric_cols].to_string(index=False))


if __name__ == "__main__":
    main()
