#!/usr/bin/env bash
set -euo pipefail

cd /home/tahiti/ARTeccv

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

PYTHON_BIN="python"
command -v python >/dev/null 2>&1 || PYTHON_BIN="python3"

mkdir -p src outputs/paper_figures

cat > src/make_gala_paper_visuals.py <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import gc
import json
import glob
import math
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")


# ============================================================
# Config
# ============================================================

OUT = Path("outputs/paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

DPI = 450
SEED = 42
KS = [1, 5, 10]
MAP_K = 10
BOOT_N = 1000

TARGETS_MAIN = ["artist_style", "style_genre", "artist"]
TARGETS_ALL = ["artist", "artist_style", "genre", "label", "style", "style_genre"]

SYSTEMS_MAIN = [
    "baseline_random_fixed",
    "baseline_clip_image",
    "llava_onevision_0_5b__full",
    "llava_onevision_0_5b__visual_only",
    "llava_onevision_0_5b__no_style",
    "llava_onevision_0_5b__no_uncertainty",
]

SYSTEMS_EXTENDED = [
    "baseline_random_fixed",
    "baseline_clip_image",
    "baseline_metadata_label_text",
    "qwen25vl_3b__full",
    "qwen25vl_3b__visual_only",
    "qwen25vl_3b__no_style",
    "qwen25vl_3b__no_uncertainty",
    "qwen25vl_3b__compact",
    "llava_onevision_0_5b__full",
    "llava_onevision_0_5b__visual_only",
    "llava_onevision_0_5b__no_style",
    "llava_onevision_0_5b__no_uncertainty",
    "llava_onevision_0_5b__compact",
]

COLORS = {
    "Random": "#9CA3AF",
    "CLIP image": "#2563EB",
    "Metadata text": "#7C3AED",
    "Qwen full": "#F59E0B",
    "Qwen visual-only": "#FBBF24",
    "Qwen no-style": "#D97706",
    "Qwen no-uncertainty": "#92400E",
    "Qwen compact": "#B45309",
    "LLaVA full": "#10B981",
    "LLaVA visual-only": "#059669",
    "LLaVA no-style": "#34D399",
    "LLaVA no-uncertainty": "#047857",
    "LLaVA compact": "#065F46",
}

TARGET_PRETTY = {
    "artist_style": "Artist--Style",
    "style_genre": "Style--Genre",
    "artist": "Artist",
    "style": "Style",
    "genre": "Genre",
    "label": "Label",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
})


# ============================================================
# Helpers
# ============================================================

def log(x):
    print(x, flush=True)


def safe_mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def savefig(fig, name):
    png = OUT / f"{name}.png"
    pdf = OUT / f"{name}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"saved: {png}")
    log(f"saved: {pdf}")


def pretty_system(s):
    s = str(s)
    mp = {
        "baseline_random_fixed": "Random",
        "baseline_clip_image": "CLIP image",
        "baseline_metadata_label_text": "Metadata text",
        "qwen25vl_3b__full": "Qwen full",
        "qwen25vl_3b__visual_only": "Qwen visual-only",
        "qwen25vl_3b__no_style": "Qwen no-style",
        "qwen25vl_3b__no_uncertainty": "Qwen no-uncertainty",
        "qwen25vl_3b__compact": "Qwen compact",
        "llava_onevision_0_5b__full": "LLaVA full",
        "llava_onevision_0_5b__visual_only": "LLaVA visual-only",
        "llava_onevision_0_5b__no_style": "LLaVA no-style",
        "llava_onevision_0_5b__no_uncertainty": "LLaVA no-uncertainty",
        "llava_onevision_0_5b__compact": "LLaVA compact",
    }
    if s in mp:
        return mp[s]
    return (
        s.replace("baseline_", "")
         .replace("qwen25vl_3b", "Qwen")
         .replace("llava_onevision_0_5b", "LLaVA")
         .replace("__", " ")
         .replace("_", "-")
    )


def pretty_target(t):
    return TARGET_PRETTY.get(str(t), str(t).replace("_", "--").title())


def norm_label(x):
    if x is None:
        return ""
    try:
        if isinstance(x, float) and np.isnan(x):
            return ""
    except Exception:
        pass
    s = str(x).strip().lower()
    s = s.replace("\\", "/")
    s = re.sub(r"\.[a-z0-9]+$", "", s)
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    bad = {"", "nan", "none", "null", "unknown", "unk", "na", "n/a", "other", "misc"}
    return "" if s in bad else s


def short_label(x, max_len=32):
    s = str(x).replace("__", " / ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[:max_len - 1] + "…"
    return s.title()


# ============================================================
# Loading data
# ============================================================

def load_retrieval_metrics():
    candidates = [
        "outputs/fixed_eval_existing/retrieval_metrics_fixed.csv",
        "outputs/fixed_eval/retrieval_metrics_fixed.csv",
        "outputs/summary_metrics_retrieval_fixed.csv",
    ]

    for p in candidates:
        pp = Path(p)
        if pp.exists():
            df = pd.read_csv(pp)
            log(f"Using retrieval metrics: {pp}")
            return df

    raise FileNotFoundError("No retrieval metrics file found.")


def load_manifest():
    candidates = [
        "outputs/manifest_for_existing_generations.csv",
        "outputs/wikiart_manifest_balanced.csv",
        "outputs/wikiart_manifest_parent.csv",
        "outputs/wikiart_manifest.csv",
    ]

    for p in candidates:
        pp = Path(p)
        if pp.exists():
            df = pd.read_csv(pp)
            log(f"Using manifest: {pp}")
            return df

    raise FileNotFoundError("No manifest found.")


def normalize_claim_cols(df):
    df = df.copy()

    if "system" not in df.columns:
        for c in df.columns:
            if c.lower() in {"name", "run", "method", "model_prompt", "experiment"}:
                df = df.rename(columns={c: "system"})
                break

    colmap = {}
    for c in df.columns:
        cl = c.lower()
        if ("unsupported" in cl or "hallucination" in cl) and "rate" in cl:
            colmap[c] = "unsupported_rate"
        elif ("unsupported" in cl or "hallucination" in cl) and ("count" in cl or "num" in cl):
            colmap[c] = "unsupported_count"
        elif cl in {"avg_words", "mean_words", "words_mean", "avg_word_count"}:
            colmap[c] = "avg_words"
        elif cl in {"empty_rate", "empty_output_rate"}:
            colmap[c] = "empty_rate"

    df = df.rename(columns=colmap)
    return df


def estimate_claims_from_generations():
    text_keys = [
        "response", "generated_text", "generation", "text", "caption",
        "description", "answer", "output"
    ]

    patterns = [
        r"\bartist\b", r"\bpainted by\b", r"\bcreated by\b", r"\bcentury\b",
        r"\bperiod\b", r"\bmovement\b", r"\bschool\b", r"\bsymbolizes\b",
        r"\brepresents\b", r"\ballegory\b", r"\bmythological\b",
        r"\breligious\b", r"\bhistorical\b", r"\bRenaissance\b",
        r"\bBaroque\b", r"\bRococo\b", r"\bRomanticism\b",
        r"\bRealism\b", r"\bImpressionism\b", r"\bExpressionism\b",
        r"\bCubism\b", r"\bSymbolism\b",
    ]
    regs = [re.compile(p, flags=re.I) for p in patterns]

    rows = []
    for f in sorted(glob.glob("outputs/generations/*.jsonl")):
        system = Path(f).stem
        n = 0
        empty = 0
        words_total = 0
        claim_count = 0

        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                txt = ""
                for k in text_keys:
                    if k in obj:
                        txt = str(obj[k])
                        break

                n += 1
                words = txt.split()
                words_total += len(words)
                if len(words) == 0:
                    empty += 1

                c = 0
                for rg in regs:
                    if rg.search(txt):
                        c += 1
                claim_count += c

        if n > 0:
            rows.append({
                "system": system,
                "unsupported_count": claim_count / n,
                "unsupported_rate": claim_count / max(words_total, 1),
                "empty_rate": empty / n,
                "avg_words": words_total / n,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "estimated_claim_metrics_from_generations.csv", index=False)
    return df


def load_claim_metrics():
    candidates = []
    candidates += ["outputs/summary_metrics.csv", "summary_metrics.csv"]
    candidates += sorted(glob.glob("outputs/backups/summary_metrics.csv*.bak"))
    candidates += sorted(glob.glob("outputs/backups/*summary_metrics*.csv*"))

    for p in candidates:
        pp = Path(p)
        if not pp.exists():
            continue
        try:
            df = pd.read_csv(pp)
            df = normalize_claim_cols(df)
            if "system" in df.columns and any(c in df.columns for c in ["unsupported_rate", "unsupported_count"]):
                log(f"Using claim metrics: {pp}")
                return df
        except Exception:
            pass

    log("Claim metrics not found; estimating from generation JSONL files.")
    return estimate_claims_from_generations()


def find_npz_by_stem(stem):
    files = sorted(glob.glob("outputs/embeddings/*.npz"))
    for f in files:
        if Path(f).stem == stem:
            return Path(f)
    for f in files:
        if stem.lower() in Path(f).stem.lower():
            return Path(f)
    return None


def load_npz_embedding(stem):
    p = find_npz_by_stem(stem)
    if p is None:
        log(f"missing embedding: {stem}")
        return None

    z = np.load(p, allow_pickle=True)
    best_key = None
    best_arr = None

    preferred = [
        "embeddings", "embedding", "vectors", "X", "x", "features", "feats",
        "image_embeds", "text_embeds"
    ]

    for k in preferred + list(z.files):
        if k not in z.files:
            continue
        arr = z[k]
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
            best_key = k
            best_arr = arr.astype(np.float32)
            break

    if best_arr is None:
        log(f"no 2D numeric array in {p}")
        return None

    log(f"loaded embedding: {p.name} key={best_key} shape={best_arr.shape}")
    return best_arr


def l2_normalize(X):
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + 1e-12)


def cosine_similarity_matrix(X):
    X = l2_normalize(X)
    return (X @ X.T).astype(np.float32)


# ============================================================
# Retrieval metrics from similarity
# ============================================================

def valid_label_mask(labels, min_count=2):
    labels = pd.Series(labels).fillna("").map(norm_label)
    vc = labels.value_counts()
    good = set(vc[vc >= min_count].index)
    mask = labels.isin(good).values
    return mask, labels.values


def query_level_metrics_from_similarity(S, labels, ks=(1, 5, 10), map_k=10):
    labels = np.asarray(labels)
    n = len(labels)
    max_k = max(max(ks), map_k)

    hits = {k: [] for k in ks}
    aps = []

    for i in range(n):
        rel = labels == labels[i]
        rel[i] = False
        total_rel = int(rel.sum())
        if total_rel <= 0:
            continue

        row = S[i].copy()
        row[i] = -np.inf

        top = np.argpartition(-row, kth=min(max_k, n - 1) - 1)[:min(max_k, n - 1)]
        top = top[np.argsort(-row[top])]
        top_rel = rel[top]

        for k in ks:
            hits[k].append(float(top_rel[:k].any()))

        h = 0
        precs = []
        for rank, is_rel in enumerate(top_rel[:map_k], start=1):
            if is_rel:
                h += 1
                precs.append(h / rank)
        denom = min(total_rel, map_k)
        aps.append(float(np.sum(precs) / denom) if denom > 0 else np.nan)

    out = {f"recall@{k}": np.asarray(v, dtype=np.float32) for k, v in hits.items()}
    out[f"map@{map_k}"] = np.asarray(aps, dtype=np.float32)
    return out


def aggregate_query_metrics(qm):
    return {k: float(np.nanmean(v)) if len(v) else np.nan for k, v in qm.items()}


def bootstrap_ci(vals, n_boot=1000, seed=42):
    vals = np.asarray(vals, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    means = []
    n = len(vals)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(float(np.mean(vals[idx])))

    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)


def random_query_metrics(labels, ks=(1, 5, 10), map_k=10, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n = len(labels)
    max_k = max(max(ks), map_k)

    hits = {k: [] for k in ks}
    aps = []

    for i in range(n):
        rel = labels == labels[i]
        rel[i] = False
        total_rel = int(rel.sum())
        if total_rel <= 0:
            continue

        candidates = np.array([j for j in range(n) if j != i])
        rng.shuffle(candidates)
        top = candidates[:max_k]
        top_rel = rel[top]

        for k in ks:
            hits[k].append(float(top_rel[:k].any()))

        h = 0
        precs = []
        for rank, is_rel in enumerate(top_rel[:map_k], start=1):
            if is_rel:
                h += 1
                precs.append(h / rank)
        denom = min(total_rel, map_k)
        aps.append(float(np.sum(precs) / denom) if denom > 0 else np.nan)

    out = {f"recall@{k}": np.asarray(v, dtype=np.float32) for k, v in hits.items()}
    out[f"map@{map_k}"] = np.asarray(aps, dtype=np.float32)
    return out


# ============================================================
# Figure 1: retrieval heatmap
# ============================================================

def make_retrieval_heatmap(metrics):
    df = metrics.copy()
    df = df[df["label_col"].isin(TARGETS_MAIN)]
    df = df[df["system"].isin(SYSTEMS_EXTENDED)]

    rows = [s for s in SYSTEMS_EXTENDED if s in set(df["system"])]
    cols = TARGETS_MAIN

    M = np.full((len(rows), len(cols)), np.nan, dtype=np.float32)

    for i, s in enumerate(rows):
        for j, t in enumerate(cols):
            sub = df[(df["system"] == s) & (df["label_col"] == t)]
            if len(sub):
                M[i, j] = float(sub.iloc[0]["recall@1"])

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=np.nanmax(M) * 1.05)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([pretty_target(c) for c in cols], fontweight="bold")

    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([pretty_system(r) for r in rows])

    for i in range(len(rows)):
        for j in range(len(cols)):
            val = M[i, j]
            if np.isfinite(val):
                color = "white" if val > np.nanmax(M) * 0.45 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        color=color, fontsize=8.5, fontweight="bold")

    ax.set_title("Retrieval strength across artwork metadata targets", pad=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Recall@1", rotation=90)

    ax.set_xlabel("Retrieval target")
    ax.set_ylabel("Representation")

    fig.tight_layout()
    savefig(fig, "fig1_retrieval_heatmap")


# ============================================================
# Figure 2: grouped retrieval dashboard
# ============================================================

def make_retrieval_dashboard(metrics):
    df = metrics.copy()
    df = df[df["label_col"].isin(TARGETS_MAIN)]
    systems = [s for s in SYSTEMS_MAIN if s in set(df["system"])]

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8), gridspec_kw={"wspace": 0.22})

    for ax, metric, title in zip(
        axes,
        ["recall@1", "recall@10"],
        ["Fine-grained hit at rank 1", "Coverage within top 10"]
    ):
        x = np.arange(len(TARGETS_MAIN))
        width = 0.12

        for k, s in enumerate(systems):
            vals = []
            for t in TARGETS_MAIN:
                sub = df[(df["system"] == s) & (df["label_col"] == t)]
                vals.append(float(sub.iloc[0][metric]) if len(sub) else np.nan)

            label = pretty_system(s)
            offset = (k - (len(systems) - 1) / 2) * width
            ax.bar(
                x + offset, vals, width=width,
                label=label,
                color=COLORS.get(label, "#999999"),
                edgecolor="white",
                linewidth=0.8
            )

        ax.set_xticks(x)
        ax.set_xticklabels([pretty_target(t) for t in TARGETS_MAIN], fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(metric.replace("recall@", "Recall@"))
        ax.grid(axis="y", alpha=0.22, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=3, frameon=False)

    fig.suptitle("Corrected retrieval evaluation: CLIP remains strongest, VLM descriptions remain useful",
                 y=1.16, fontweight="bold")

    fig.tight_layout()
    savefig(fig, "fig2_retrieval_dashboard")


# ============================================================
# Figure 3: reliability/retrieval tradeoff
# ============================================================

def make_tradeoff_scatter(metrics, claims):
    claims = claims.copy()
    claims = normalize_claim_cols(claims)

    if "system" not in claims.columns:
        log("skip tradeoff: no system in claim metrics")
        return

    target = "artist_style"
    ret = metrics[(metrics["label_col"] == target)].copy()
    ret = ret[~ret["system"].astype(str).str.contains("random|clip|metadata", case=False, na=False)]

    merged = ret.merge(claims, on="system", how="left")

    if "unsupported_rate" not in merged.columns:
        log("skip tradeoff: no unsupported_rate")
        return

    merged = merged[np.isfinite(merged["unsupported_rate"])].copy()
    if len(merged) == 0:
        log("skip tradeoff: empty after filtering")
        return

    merged["pretty"] = merged["system"].map(pretty_system)

    fig, ax = plt.subplots(figsize=(7.4, 5.6))

    for _, r in merged.iterrows():
        name = r["pretty"]
        color = COLORS.get(name, "#444444")
        marker = "o" if "LLaVA" in name else "s"

        ax.scatter(
            r["unsupported_rate"],
            r["recall@1"],
            s=120,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            marker=marker,
            zorder=3
        )

        ax.text(
            r["unsupported_rate"] + 0.002,
            r["recall@1"] + 0.003,
            name.replace("Qwen ", "Q-").replace("LLaVA ", "L-"),
            fontsize=8,
            color="#111827"
        )

    ax.set_xlabel("Unsupported-claim proxy rate ↓")
    ax.set_ylabel("Artist--Style Recall@1 ↑")
    ax.set_title("Retrieval--reliability trade-off", fontweight="bold", pad=12)
    ax.grid(alpha=0.22, linestyle="--")

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.annotate(
        "safer",
        xy=(xmin + 0.05 * (xmax - xmin), ymin + 0.08 * (ymax - ymin)),
        xytext=(xmin + 0.30 * (xmax - xmin), ymin + 0.08 * (ymax - ymin)),
        arrowprops=dict(arrowstyle="->", color="#374151", lw=1.2),
        color="#374151",
        fontsize=9,
        va="center"
    )

    ax.annotate(
        "better retrieval",
        xy=(xmin + 0.06 * (xmax - xmin), ymax - 0.05 * (ymax - ymin)),
        xytext=(xmin + 0.06 * (xmax - xmin), ymax - 0.30 * (ymax - ymin)),
        arrowprops=dict(arrowstyle="->", color="#374151", lw=1.2),
        color="#374151",
        fontsize=9,
        ha="center"
    )

    merged.to_csv(OUT / "tradeoff_retrieval_reliability.csv", index=False)
    savefig(fig, "fig3_retrieval_reliability_tradeoff")


# ============================================================
# Figure 4: unsupported-claim ablation bars
# ============================================================

def make_safety_ablation_bars(claims):
    claims = normalize_claim_cols(claims)
    if "system" not in claims.columns or "unsupported_rate" not in claims.columns:
        log("skip safety ablation: required columns missing")
        return

    prompt_order = ["full", "visual_only", "no_style", "no_uncertainty", "compact"]

    rows = []
    for _, r in claims.iterrows():
        s = str(r["system"])
        model = None
        prompt = None
        if s.startswith("qwen25vl_3b__"):
            model = "Qwen2.5-VL"
            prompt = s.split("__", 1)[1]
        elif s.startswith("llava_onevision_0_5b__"):
            model = "LLaVA-OneVision"
            prompt = s.split("__", 1)[1]
        if model and prompt in prompt_order:
            rows.append({
                "model": model,
                "prompt": prompt,
                "unsupported_rate": float(r["unsupported_rate"]),
                "unsupported_count": float(r["unsupported_count"]) if "unsupported_count" in claims.columns and pd.notna(r.get("unsupported_count", np.nan)) else np.nan,
                "empty_rate": float(r["empty_rate"]) if "empty_rate" in claims.columns and pd.notna(r.get("empty_rate", np.nan)) else np.nan,
            })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        log("skip safety ablation: no qwen/llava rows")
        return

    df.to_csv(OUT / "safety_ablation_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    x = np.arange(len(prompt_order))
    width = 0.34

    for i, model in enumerate(["Qwen2.5-VL", "LLaVA-OneVision"]):
        vals = []
        for p in prompt_order:
            sub = df[(df["model"] == model) & (df["prompt"] == p)]
            vals.append(float(sub.iloc[0]["unsupported_rate"]) if len(sub) else np.nan)

        color = "#F59E0B" if model.startswith("Qwen") else "#10B981"
        offset = -width / 2 if i == 0 else width / 2

        ax.bar(
            x + offset,
            vals,
            width=width,
            label=model,
            color=color,
            edgecolor="white",
            linewidth=1.0
        )

        for xi, yi in zip(x + offset, vals):
            if np.isfinite(yi):
                ax.text(xi, yi + max(vals) * 0.03, f"{yi:.3f}",
                        ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["full", "visual\nonly", "no\nstyle", "no\nuncertainty", "compact"],
        fontweight="bold"
    )
    ax.set_ylabel("Unsupported-claim proxy rate ↓")
    ax.set_title("Prompt constraints reduce unsupported art-historical claims", fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    savefig(fig, "fig4_prompt_safety_ablation")


# ============================================================
# Figure 5: hybrid CLIP + VLM curves
# ============================================================

def make_hybrid_curves(manifest):
    clip = load_npz_embedding("baseline_clip_image")
    vlm = load_npz_embedding("llava_onevision_0_5b__visual_only")

    if clip is None or vlm is None:
        log("skip hybrid curves: missing CLIP or LLaVA visual-only embeddings")
        return

    n = min(len(manifest), clip.shape[0], vlm.shape[0])
    manifest = manifest.iloc[:n].copy()
    clip = clip[:n]
    vlm = vlm[:n]

    log("computing CLIP similarity")
    S_clip = cosine_similarity_matrix(clip)
    del clip
    gc.collect()

    log("computing LLaVA visual-only similarity")
    S_vlm = cosine_similarity_matrix(vlm)
    del vlm
    gc.collect()

    alphas = np.linspace(0, 1, 11)
    rows = []

    for target in TARGETS_MAIN:
        if target not in manifest.columns:
            continue

        mask, labels = valid_label_mask(manifest[target].values, min_count=2)
        idx = np.where(mask)[0]
        if len(idx) < 10:
            continue

        labels_sub = labels[idx]
        S_clip_sub = S_clip[np.ix_(idx, idx)]
        S_vlm_sub = S_vlm[np.ix_(idx, idx)]

        for a in alphas:
            S = (a * S_clip_sub + (1 - a) * S_vlm_sub).astype(np.float32)
            qm = query_level_metrics_from_similarity(S, labels_sub, ks=KS, map_k=MAP_K)
            ag = aggregate_query_metrics(qm)
            ag.update({
                "target": target,
                "alpha_clip": float(a),
                "alpha_vlm": float(1 - a),
                "n_items": int(len(labels_sub)),
                "n_classes": int(pd.Series(labels_sub).nunique()),
            })
            rows.append(ag)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "hybrid_clip_vlm_curves.csv", index=False)

    if len(df) == 0:
        log("skip hybrid plot: no rows")
        return

    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    for target, color in zip(TARGETS_MAIN, ["#2563EB", "#10B981", "#F59E0B"]):
        sub = df[df["target"] == target].sort_values("alpha_clip")
        if len(sub) == 0:
            continue
        ax.plot(
            sub["alpha_clip"],
            sub["recall@1"],
            marker="o",
            linewidth=2.4,
            markersize=6,
            label=pretty_target(target),
            color=color
        )

    ax.set_xlabel(r"Hybrid weight $\alpha$ for CLIP image similarity")
    ax.set_ylabel("Recall@1")
    ax.set_title(r"Hybrid retrieval: $S=\alpha S_{\mathrm{CLIP}}+(1-\alpha)S_{\mathrm{VLM}}$",
                 fontweight="bold", pad=12)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False)

    ax.text(0.00, ax.get_ylim()[0], "VLM only", ha="left", va="bottom", fontsize=9, color="#374151")
    ax.text(1.00, ax.get_ylim()[0], "CLIP only", ha="right", va="bottom", fontsize=9, color="#374151")

    fig.tight_layout()
    savefig(fig, "fig5_hybrid_retrieval_curves")

    del S_clip
    del S_vlm
    gc.collect()


# ============================================================
# Figure 6: qualitative retrieval gallery
# ============================================================

def load_square_image(path, size=160):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        x = (size - img.width) // 2
        y = (size - img.height) // 2
        canvas.paste(img, (x, y))
        return np.asarray(canvas)
    except Exception:
        canvas = Image.new("RGB", (size, size), "#F3F4F6")
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, size - 1, size - 1], outline="#9CA3AF", width=3)
        draw.text((18, size // 2 - 8), "missing", fill="#374151")
        return np.asarray(canvas)


def topk_indices(S, i, k=4):
    row = S[i].copy()
    row[i] = -np.inf
    top = np.argpartition(-row, kth=k - 1)[:k]
    top = top[np.argsort(-row[top])]
    return top.tolist()


def make_qualitative_gallery(manifest):
    if "image_path" not in manifest.columns:
        log("skip qualitative gallery: no image_path in manifest")
        return

    clip = load_npz_embedding("baseline_clip_image")
    vlm = load_npz_embedding("llava_onevision_0_5b__visual_only")

    if clip is None or vlm is None:
        log("skip qualitative gallery: missing embeddings")
        return

    target = "artist_style" if "artist_style" in manifest.columns else "style_genre"
    if target not in manifest.columns:
        log("skip qualitative gallery: no target column")
        return

    n = min(len(manifest), clip.shape[0], vlm.shape[0])
    manifest = manifest.iloc[:n].copy()
    clip = clip[:n]
    vlm = vlm[:n]

    labels = manifest[target].fillna("").astype(str).values
    vc = pd.Series(labels).value_counts()
    good_labels = set(vc[(vc >= 6) & (vc <= 120)].index)

    mask = np.array([x in good_labels and Path(str(p)).exists()
                     for x, p in zip(labels, manifest["image_path"].values)])
    idx_pool = np.where(mask)[0]

    if len(idx_pool) < 10:
        log("skip qualitative gallery: not enough valid images")
        return

    log("computing similarities for qualitative gallery")
    S_clip = cosine_similarity_matrix(clip)
    S_vlm = cosine_similarity_matrix(vlm)

    candidates = []
    for i in idx_pool[::max(1, len(idx_pool) // 600)]:
        tc = topk_indices(S_clip, i, k=4)
        tv = topk_indices(S_vlm, i, k=4)
        score = sum(labels[j] == labels[i] for j in tc) + sum(labels[j] == labels[i] for j in tv)
        candidates.append((score, i))

    candidates = sorted(candidates, reverse=True)

    chosen = []
    used_labels = set()
    for score, i in candidates:
        if labels[i] in used_labels:
            continue
        chosen.append(i)
        used_labels.add(labels[i])
        if len(chosen) >= 4:
            break

    if len(chosen) == 0:
        chosen = idx_pool[:4].tolist()

    K = 4
    n_rows = len(chosen)
    n_cols = 1 + K + K

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15.6, 2.55 * n_rows))

    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = ["Query"] + [f"CLIP #{i}" for i in range(1, K + 1)] + [f"LLaVA #{i}" for i in range(1, K + 1)]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontweight="bold", fontsize=10)

    for r, qi in enumerate(chosen):
        clip_top = topk_indices(S_clip, qi, k=K)
        vlm_top = topk_indices(S_vlm, qi, k=K)
        row_indices = [qi] + clip_top + vlm_top

        for c, idx in enumerate(row_indices):
            ax = axes[r, c]
            ax.imshow(load_square_image(manifest.iloc[idx]["image_path"], size=180))
            ax.set_xticks([])
            ax.set_yticks([])

            is_query = c == 0
            is_match = labels[idx] == labels[qi]

            if is_query:
                edge = "#2563EB"
                lw = 4
            elif is_match:
                edge = "#10B981"
                lw = 4
            else:
                edge = "#EF4444"
                lw = 3

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(edge)
                spine.set_linewidth(lw)

            lab = short_label(labels[idx], 28)
            ax.text(
                0.5, -0.08, lab,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=7.4,
                color="#111827",
                wrap=True
            )

        axes[r, 0].text(
            -0.20, 0.5, f"Q{r+1}",
            transform=axes[r, 0].transAxes,
            ha="right", va="center",
            fontsize=13,
            fontweight="bold",
            color="#111827"
        )

    legend_elements = [
        Line2D([0], [0], color="#2563EB", lw=4, label="Query"),
        Line2D([0], [0], color="#10B981", lw=4, label="same target"),
        Line2D([0], [0], color="#EF4444", lw=4, label="different target"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=3, frameon=False)

    fig.suptitle("Qualitative retrieval examples: visual CLIP baseline and VLM-description retrieval",
                 y=1.055, fontweight="bold")

    fig.tight_layout()
    savefig(fig, "fig6_qualitative_retrieval_gallery")

    del S_clip
    del S_vlm
    gc.collect()


# ============================================================
# Figure 7: bootstrap CI table
# ============================================================

def make_bootstrap_ci_tables(manifest):
    systems = [
        "baseline_clip_image",
        "llava_onevision_0_5b__visual_only",
        "llava_onevision_0_5b__no_uncertainty",
        "llava_onevision_0_5b__full",
    ]

    rows = []

    for system in systems:
        X = load_npz_embedding(system)
        if X is None:
            continue

        n = min(len(manifest), X.shape[0])
        m = manifest.iloc[:n].copy()
        X = X[:n]

        log(f"bootstrap CI: computing S for {system}")
        S = cosine_similarity_matrix(X)

        for target in TARGETS_MAIN:
            if target not in m.columns:
                continue

            mask, labels = valid_label_mask(m[target].values, min_count=2)
            idx = np.where(mask)[0]
            if len(idx) < 10:
                continue

            labels_sub = labels[idx]
            S_sub = S[np.ix_(idx, idx)]
            qm = query_level_metrics_from_similarity(S_sub, labels_sub, ks=KS, map_k=MAP_K)

            for metric, vals in qm.items():
                mean, lo, hi = bootstrap_ci(vals, n_boot=BOOT_N, seed=SEED)
                rows.append({
                    "target": target,
                    "system": system,
                    "metric": metric,
                    "mean": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_queries": len(vals),
                    "n_classes": int(pd.Series(labels_sub).nunique()),
                })

        del S
        del X
        gc.collect()

    # random baseline
    for target in TARGETS_MAIN:
        if target not in manifest.columns:
            continue
        mask, labels = valid_label_mask(manifest[target].values, min_count=2)
        labels_sub = labels[mask]
        qm = random_query_metrics(labels_sub, ks=KS, map_k=MAP_K, seed=SEED)
        for metric, vals in qm.items():
            mean, lo, hi = bootstrap_ci(vals, n_boot=BOOT_N, seed=SEED)
            rows.append({
                "target": target,
                "system": "baseline_random_fixed",
                "metric": metric,
                "mean": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "n_queries": len(vals),
                "n_classes": int(pd.Series(labels_sub).nunique()),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bootstrap_ci_query_level.csv", index=False)

    # compact tex: Recall@1 only
    r1 = df[df["metric"] == "recall@1"].copy()
    r1["System"] = r1["system"].map(pretty_system)
    r1["Target"] = r1["target"].map(pretty_target)
    r1["R@1 95% CI"] = r1.apply(
        lambda r: f"{r['mean']:.3f} [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]",
        axis=1
    )
    tex = r1[["Target", "System", "R@1 95% CI", "n_classes"]].to_latex(
        index=False,
        escape=False,
        caption="Query-level bootstrap confidence intervals for Recall@1.",
        label="tab:bootstrap_ci_recall1"
    )
    (OUT / "bootstrap_ci_recall1.tex").write_text(tex, encoding="utf-8")

    log(f"saved: {OUT / 'bootstrap_ci_query_level.csv'}")
    log(f"saved: {OUT / 'bootstrap_ci_recall1.tex'}")


# ============================================================
# Bonus: save all key tables as TeX
# ============================================================

def save_paper_tables(metrics, claims):
    # Main retrieval table
    df = metrics.copy()
    keep = [
        "baseline_random_fixed",
        "baseline_clip_image",
        "llava_onevision_0_5b__full",
        "llava_onevision_0_5b__visual_only",
        "llava_onevision_0_5b__no_uncertainty",
        "llava_onevision_0_5b__no_style",
    ]
    tab = df[df["label_col"].isin(TARGETS_MAIN) & df["system"].isin(keep)].copy()
    tab["Target"] = tab["label_col"].map(pretty_target)
    tab["System"] = tab["system"].map(pretty_system)

    order_t = {t: i for i, t in enumerate(TARGETS_MAIN)}
    order_s = {s: i for i, s in enumerate(keep)}
    tab["ot"] = tab["label_col"].map(order_t)
    tab["os"] = tab["system"].map(order_s)
    tab = tab.sort_values(["ot", "os"])

    out = tab[["Target", "System", "recall@1", "recall@5", "recall@10", "map@10"]].copy()
    out.columns = ["Target", "System", "R@1", "R@5", "R@10", "mAP@10"]
    out.to_csv(OUT / "paper_main_retrieval_table.csv", index=False)
    tex = out.to_latex(
        index=False,
        escape=False,
        float_format=lambda x: f"{x:.3f}",
        caption="Corrected retrieval performance under non-degenerate artwork metadata targets.",
        label="tab:gala_retrieval_main"
    )
    (OUT / "paper_main_retrieval_table.tex").write_text(tex, encoding="utf-8")

    # Tradeoff table
    claims = normalize_claim_cols(claims)
    if "system" in claims.columns and "unsupported_rate" in claims.columns:
        ret = metrics[metrics["label_col"] == "artist_style"].copy()
        ret = ret[~ret["system"].str.contains("random|clip|metadata", case=False, na=False)]
        m = ret.merge(claims, on="system", how="left")
        cols = ["system", "recall@1", "recall@10", "map@10", "unsupported_rate", "unsupported_count", "empty_rate", "avg_words"]
        cols = [c for c in cols if c in m.columns]
        m = m[cols].copy()
        m["System"] = m["system"].map(pretty_system)
        first = ["System"] + [c for c in cols if c != "system"]
        m = m[first]
        m.to_csv(OUT / "paper_tradeoff_table.csv", index=False)
        tex = m.to_latex(
            index=False,
            escape=False,
            float_format=lambda x: f"{x:.3f}",
            caption="Retrieval--reliability trade-off for generated artwork descriptions.",
            label="tab:gala_tradeoff"
        )
        (OUT / "paper_tradeoff_table.tex").write_text(tex, encoding="utf-8")

    log(f"saved: {OUT / 'paper_main_retrieval_table.tex'}")
    log(f"saved: {OUT / 'paper_tradeoff_table.tex'}")


# ============================================================
# Main
# ============================================================

def main():
    log("============================================================")
    log("GALA paper visual generation")
    log("============================================================")

    metrics = load_retrieval_metrics()
    manifest = load_manifest()
    claims = load_claim_metrics()

    log("\n[1/7] Retrieval heatmap")
    make_retrieval_heatmap(metrics)

    log("\n[2/7] Retrieval dashboard")
    make_retrieval_dashboard(metrics)

    log("\n[3/7] Retrieval--reliability tradeoff")
    make_tradeoff_scatter(metrics, claims)

    log("\n[4/7] Prompt safety ablation")
    make_safety_ablation_bars(claims)

    log("\n[5/7] Hybrid CLIP+VLM curves")
    make_hybrid_curves(manifest)

    log("\n[6/7] Qualitative retrieval gallery")
    make_qualitative_gallery(manifest)

    log("\n[7/7] Bootstrap CI and paper tables")
    make_bootstrap_ci_tables(manifest)
    save_paper_tables(metrics, claims)

    log("\n============================================================")
    log("DONE. Figures saved in outputs/paper_figures/")
    log("============================================================")
    for f in sorted(OUT.glob("*")):
        log(f"{f}  {f.stat().st_size/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()
PY

$PYTHON_BIN -u src/make_gala_paper_visuals.py 2>&1 | tee outputs/paper_figures/make_gala_paper_visuals.log

echo ""
echo "READY FIGURES:"
ls -lh outputs/paper_figures | sed -n '1,200p'
