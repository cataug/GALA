import argparse
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from art_common import hallucination_proxy, lexical_contains_label, load_config, read_jsonl


def load_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    ids = z["image_ids"].astype(str)
    labels = z["labels"].astype(str)
    emb = z["embeddings"].astype("float32")
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    return ids, labels, emb


def average_precision_at_k(relevant: np.ndarray, k: int) -> float:
    relevant = relevant[:k].astype(bool)
    if relevant.sum() == 0:
        return 0.0
    precisions = []
    hits = 0
    for i, rel in enumerate(relevant, start=1):
        if rel:
            hits += 1
            precisions.append(hits / i)
    return float(np.mean(precisions)) if precisions else 0.0


def eval_retrieval(path: Path, ks: List[int]) -> Dict[str, float]:
    ids, labels, emb = load_npz(path)

    sim = emb @ emb.T
    np.fill_diagonal(sim, -1e9)

    order = np.argsort(-sim, axis=1)
    metrics = {}

    for k in ks:
        hits = []
        ap = []
        for i in range(len(ids)):
            top = order[i, :k]
            rel = labels[top] == labels[i]
            hits.append(float(rel.any()))
            ap.append(average_precision_at_k(rel, k))
        metrics[f"recall@{k}"] = float(np.mean(hits))
        metrics[f"map@{k}"] = float(np.mean(ap))

    metrics["n_items"] = int(len(ids))
    metrics["n_classes"] = int(len(set(labels.tolist())))
    return metrics


def eval_generation_jsonl(path: Path) -> Dict[str, float]:
    rows = read_jsonl(path)
    rows = [r for r in rows if r.get("response") is not None]
    if not rows:
        return {}

    lens = []
    lex = []
    risk_rates = []
    risk_counts = []
    empty = 0

    for r in rows:
        txt = r.get("response", "") or ""
        label = r.get("label", "") or ""
        if not txt.strip():
            empty += 1
        toks = txt.split()
        lens.append(len(toks))
        lex.append(lexical_contains_label(txt, label))
        hp = hallucination_proxy(txt, label)
        risk_rates.append(float(hp["risk_rate"]))
        risk_counts.append(float(hp["risk_count"]))

    return {
        "n_generations": len(rows),
        "empty_rate": empty / max(len(rows), 1),
        "avg_words": float(np.mean(lens)),
        "median_words": float(np.median(lens)),
        "lexical_label_alignment": float(np.mean(lex)),
        "hallucination_proxy_rate": float(np.mean(risk_rates)),
        "hallucination_proxy_count": float(np.mean(risk_counts)),
    }


def parse_name(path: Path) -> Dict[str, str]:
    name = path.stem
    if name.startswith("baseline_"):
        return {"system": name, "model_key": "baseline", "ablation": name.replace("baseline_", "")}
    if "__" in name:
        model_key, ablation = name.split("__", 1)
        return {"system": name, "model_key": model_key, "ablation": ablation}
    return {"system": name, "model_key": "unknown", "ablation": "unknown"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/art_method.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    outputs = Path(cfg["paths"]["outputs_dir"])
    ks = [int(k) for k in cfg["retrieval"].get("ks", [1, 5, 10])]

    rows = []

    emb_dir = outputs / "embeddings"
    for npz in sorted(emb_dir.glob("*.npz")):
        info = parse_name(npz)
        try:
            m = eval_retrieval(npz, ks)
            row = {**info, "embedding_file": str(npz), **m}
            rows.append(row)
        except Exception as e:
            print("failed retrieval eval:", npz, repr(e))

    gen_metrics = {}
    gen_dir = outputs / "generations"
    for js in sorted(gen_dir.glob("*.jsonl")):
        info = parse_name(js)
        try:
            gm = eval_generation_jsonl(js)
            gen_metrics[info["system"]] = gm
        except Exception as e:
            print("failed gen eval:", js, repr(e))

    final = []
    for r in rows:
        gm = gen_metrics.get(r["system"], {})
        final.append({**r, **gm})

    df = pd.DataFrame(final)
    out = outputs / "summary_metrics.csv"
    df.to_csv(out, index=False)

    if not df.empty:
        sort_cols = [c for c in ["recall@10", "recall@5", "map@10"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=False)

    print("saved:", out)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
