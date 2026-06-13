#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def load_metrics():
    candidates = [
        Path("outputs/fixed_eval_existing/retrieval_metrics_fixed.csv"),
        Path("outputs/summary_metrics_retrieval_fixed.csv"),
        Path("outputs/fixed_eval/retrieval_metrics_fixed.csv"),
    ]
    for p in candidates:
        if p.exists():
            print("Using retrieval metrics:", p)
            return pd.read_csv(p)
    raise FileNotFoundError("No retrieval metrics CSV found.")


def load_claim_metrics():
    candidates = [
        Path("summary_metrics.csv"),
        Path("outputs/summary_metrics.csv"),
        Path("outputs/backups/summary_metrics.csv"),
    ]

    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                print("Using claim metrics:", p)
                return df
            except Exception:
                pass

    # Fallback: estimate from generation JSONL files.
    rows = []
    claim_words = [
        "probably", "possibly", "likely", "may", "might", "suggests",
        "appears to", "could be", "seems", "period", "century",
        "artist", "school", "movement", "symbolizes", "represents"
    ]

    for f in sorted(glob.glob("outputs/generations/*.jsonl")):
        system = Path(f).stem
        n = 0
        empty = 0
        words_total = 0
        proxy_count = 0

        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = ""
                for k in ["response", "generated_text", "generation", "text", "caption", "description", "answer", "output"]:
                    if k in obj:
                        text = str(obj[k])
                        break

                n += 1
                if not text.strip():
                    empty += 1

                words = text.split()
                words_total += len(words)

                low = text.lower()
                c = sum(1 for w in claim_words if w in low)
                proxy_count += c

        rows.append({
            "system": system,
            "empty_rate": empty / max(n, 1),
            "avg_words": words_total / max(n, 1),
            "unsupported_proxy_count": proxy_count / max(n, 1),
            "unsupported_proxy_rate": proxy_count / max(words_total, 1),
        })

    df = pd.DataFrame(rows)
    print("Claim metrics estimated from generations.")
    return df


def normalize_system_name(s):
    s = str(s)
    return (
        s.replace("baseline_", "")
         .replace("qwen25vl_3b", "Qwen2.5-VL")
         .replace("llava_onevision_0_5b", "LLaVA-OneVision")
         .replace("__", " / ")
         .replace("_", "-")
    )


def make_compact_retrieval_table(df):
    targets = ["artist_style", "style_genre", "artist"]

    keep = [
        "baseline_random_fixed",
        "baseline_clip_image",
        "llava_onevision_0_5b__full",
        "llava_onevision_0_5b__visual_only",
        "llava_onevision_0_5b__no_style",
        "llava_onevision_0_5b__no_uncertainty",
        "qwen25vl_3b__full",
        "qwen25vl_3b__visual_only",
        "qwen25vl_3b__no_style",
        "qwen25vl_3b__no_uncertainty",
    ]

    tab = df[df["label_col"].isin(targets)].copy()
    tab = tab[tab["system"].isin(keep)].copy()

    target_order = {x: i for i, x in enumerate(targets)}
    system_order = {x: i for i, x in enumerate(keep)}

    tab["target_order"] = tab["label_col"].map(target_order)
    tab["system_order"] = tab["system"].map(system_order)
    tab = tab.sort_values(["target_order", "system_order"])

    out_cols = ["label_col", "system", "n_classes", "recall@1", "recall@5", "recall@10", "map@10"]
    out_cols = [c for c in out_cols if c in tab.columns]
    tab = tab[out_cols].copy()
    tab["system_print"] = tab["system"].map(normalize_system_name)

    Path("outputs/extra").mkdir(parents=True, exist_ok=True)
    tab.to_csv("outputs/extra/compact_retrieval_table.csv", index=False)

    tex = tab[["label_col", "system_print", "recall@1", "recall@5", "recall@10", "map@10"]].copy()
    tex.columns = ["Target", "System", "R@1", "R@5", "R@10", "mAP@10"]
    tex = tex.to_latex(
        index=False,
        escape=False,
        float_format=lambda x: f"{x:.3f}",
        caption="Retrieval performance on corrected non-degenerate artwork metadata targets.",
        label="tab:retrieval_corrected"
    )
    Path("outputs/extra/compact_retrieval_table.tex").write_text(tex, encoding="utf-8")

    print("Saved outputs/extra/compact_retrieval_table.csv")
    print("Saved outputs/extra/compact_retrieval_table.tex")


def make_tradeoff_table(retrieval, claims):
    if "system" not in claims.columns:
        print("No system column in claim metrics; skipping tradeoff table.")
        return

    # Normalize possible column names.
    colmap = {}
    for c in claims.columns:
        cl = c.lower()
        if "unsupported" in cl and "rate" in cl:
            colmap[c] = "unsupported_rate"
        elif "unsupported" in cl and "count" in cl:
            colmap[c] = "unsupported_count"
        elif cl in ["avg_words", "mean_words", "words"]:
            colmap[c] = "avg_words"
        elif cl in ["empty_rate", "empty"]:
            colmap[c] = "empty_rate"

    claims = claims.rename(columns=colmap)

    target = "artist_style"
    ret = retrieval[retrieval["label_col"] == target].copy()

    # Keep prompt systems only.
    ret = ret[~ret["system"].astype(str).str.contains("random|clip|metadata", case=False, na=False)].copy()

    merged = ret.merge(claims, on="system", how="left")

    cols = [
        "system", "recall@1", "recall@5", "recall@10", "map@10",
        "unsupported_rate", "unsupported_count", "empty_rate", "avg_words"
    ]
    cols = [c for c in cols if c in merged.columns]

    merged = merged[cols].sort_values(["recall@1", "map@10"], ascending=False)

    Path("outputs/extra").mkdir(parents=True, exist_ok=True)
    merged.to_csv("outputs/extra/retrieval_reliability_tradeoff.csv", index=False)

    tex = merged.copy()
    tex["system"] = tex["system"].map(normalize_system_name)
    tex = tex.rename(columns={
        "system": "System",
        "recall@1": "R@1",
        "recall@5": "R@5",
        "recall@10": "R@10",
        "map@10": "mAP@10",
        "unsupported_rate": "Unsup. rate",
        "unsupported_count": "Unsup. count",
        "empty_rate": "Empty rate",
        "avg_words": "Avg. words",
    })

    Path("outputs/extra/retrieval_reliability_tradeoff.tex").write_text(
        tex.to_latex(
            index=False,
            escape=False,
            float_format=lambda x: f"{x:.3f}",
            caption="Retrieval--reliability trade-off for generated artwork descriptions on the artist--style target.",
            label="tab:retrieval_reliability_tradeoff"
        ),
        encoding="utf-8"
    )

    print("Saved outputs/extra/retrieval_reliability_tradeoff.csv")
    print("Saved outputs/extra/retrieval_reliability_tradeoff.tex")


def bootstrap_ci_from_existing_metrics(df):
    # This is a lightweight reporting helper, not true query-level bootstrap.
    # It summarizes variability across targets for each system.
    targets = ["artist_style", "style_genre", "artist"]
    sub = df[df["label_col"].isin(targets)].copy()

    rows = []
    for system, g in sub.groupby("system"):
        for metric in ["recall@1", "recall@5", "recall@10", "map@10"]:
            if metric not in g.columns:
                continue
            vals = g[metric].dropna().values
            if len(vals) == 0:
                continue

            rng = np.random.default_rng(42)
            boots = []
            for _ in range(5000):
                sample = rng.choice(vals, size=len(vals), replace=True)
                boots.append(np.mean(sample))

            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({
                "system": system,
                "metric": metric,
                "mean_across_targets": np.mean(vals),
                "ci95_low": lo,
                "ci95_high": hi,
                "n_targets": len(vals),
            })

    out = pd.DataFrame(rows)
    out.to_csv("outputs/extra/target_bootstrap_ci.csv", index=False)

    print("Saved outputs/extra/target_bootstrap_ci.csv")


def main():
    Path("outputs/extra").mkdir(parents=True, exist_ok=True)

    retrieval = load_metrics()
    claims = load_claim_metrics()

    make_compact_retrieval_table(retrieval)
    make_tradeoff_table(retrieval, claims)
    bootstrap_ci_from_existing_metrics(retrieval)

    print("\nDONE. Extra fast experiments saved in outputs/extra/")


if __name__ == "__main__":
    main()
