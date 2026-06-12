import json
import os
import random
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, rows: List[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9а-яё_\-\.]+", "_", s, flags=re.I)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_gpu_memory() -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns used_gb, total_gb, util_ratio.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        first = out.strip().splitlines()[0]
        used_mb, total_mb = [float(x.strip()) for x in first.split(",")]
        used_gb = used_mb / 1024.0
        total_gb = total_mb / 1024.0
        return used_gb, total_gb, used_gb / max(total_gb, 1e-9)
    except Exception:
        return None, None, None


def print_gpu(prefix: str = "GPU") -> None:
    used, total, util = get_gpu_memory()
    if used is None:
        print(f"{prefix}: nvidia-smi unavailable")
    else:
        print(f"{prefix}: used={used:.2f}GB total={total:.2f}GB util={util:.3f}")


def cuda_oom(e: BaseException) -> bool:
    s = str(e).lower()
    return (
        "out of memory" in s
        or "cuda error: out of memory" in s
        or "cublas_status_alloc_failed" in s
        or "cuda out of memory" in s
    )


def clean_cuda() -> None:
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def normalize_label(s: str) -> str:
    s = s.replace("_", " ").replace("-", " ").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize_label(label: str) -> List[str]:
    label = normalize_label(label)
    toks = re.findall(r"[a-zа-яё0-9]+", label, flags=re.I)
    return [t for t in toks if len(t) > 2]


def lexical_contains_label(text: str, label: str) -> float:
    text_n = normalize_label(text)
    toks = tokenize_label(label)
    if not toks:
        return 0.0
    hit = sum(1 for t in toks if t in text_n)
    return hit / len(toks)


RISK_PATTERNS = [
    r"\bpainted by\b",
    r"\bcreated by\b",
    r"\bby [A-Z][a-z]+ [A-Z][a-z]+\b",
    r"\bartist is\b",
    r"\bthe artist\b",
    r"\b\d{4}\b",
    r"\b\d{1,2}(st|nd|rd|th) century\b",
    r"\bbelongs to the collection\b",
    r"\bmuseum\b",
    r"\btitle\b",
    r"\bnamed\b",
    r"\bwas made in\b",
    r"\bwas created in\b",
]


def hallucination_proxy(text: str, allowed_label: str = "") -> Dict[str, float | int | List[str]]:
    """
    Lightweight proxy. It does not replace human annotation.
    It marks risky factual claims likely unsupported by image alone.
    """
    if not text:
        return {"risk_count": 0, "sent_count": 0, "risk_rate": 0.0, "hits": []}

    allowed = normalize_label(allowed_label)
    hits = []
    for pat in RISK_PATTERNS:
        for m in re.finditer(pat, text, flags=re.I):
            phrase = m.group(0)
            if allowed and normalize_label(phrase) in allowed:
                continue
            hits.append(phrase)

    sents = [s for s in re.split(r"[.!?;\n]+", text) if s.strip()]
    risk_count = len(hits)
    sent_count = max(len(sents), 1)
    return {
        "risk_count": risk_count,
        "sent_count": sent_count,
        "risk_rate": risk_count / sent_count,
        "hits": hits[:20],
    }
