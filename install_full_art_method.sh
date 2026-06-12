#!/usr/bin/env bash
set -euo pipefail

cd /home/tahiti/ARTeccv

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

pip install --no-cache-dir \
  scikit-learn \
  scipy \
  joblib \
  pyyaml \
  rich \
  sentence-transformers \
  open_clip_torch \
  faiss-cpu

mkdir -p src scripts configs outputs logs runs

cat > configs/art_method.yaml <<'YAML'
project_name: ARTeccv

paths:
  project_dir: /home/tahiti/ARTeccv
  wikiart_dir: /home/tahiti/ARTeccv/data/wikiart
  models_dir: /home/tahiti/ARTeccv/models
  outputs_dir: /home/tahiti/ARTeccv/outputs
  runs_dir: /home/tahiti/ARTeccv/runs

data:
  image_extensions: [".jpg", ".jpeg", ".png", ".webp"]
  label_mode: parent
  max_images: null
  min_per_class: 2
  seed: 42

generation:
  max_new_tokens: 220
  temperature: 0.0
  do_sample: false
  auto_batch: true
  start_batch_qwen: 8
  start_batch_llava: 16
  max_batch_qwen: 32
  max_batch_llava: 64
  increase_every_success_batches: 8
  target_gpu_mem_util: 0.92

models:
  qwen25vl_3b:
    kind: qwen25vl
    local_dir: /home/tahiti/ARTeccv/models/qwen2_5_vl_3b_instruct

  llava_onevision_0_5b:
    kind: llava_onevision
    local_dir: /home/tahiti/ARTeccv/models/llava_onevision_qwen2_0_5b_ov_hf

ablations:
  full: >
    Describe this artwork for an art-history retrieval system.
    Use the following sections exactly:
    Visible content:
    Composition:
    Color and texture:
    Style cues:
    Uncertain hypotheses:
    Mention visible objects, figures, scene elements, spatial layout, color palette, brushwork or texture cues, and possible period or style cues.
    Separate directly visible evidence from uncertain interpretation.
    Do not invent the artist name, exact title, exact date, or museum.

  visual_only: >
    Describe only what is directly visible in this artwork.
    Use the following sections exactly:
    Visible content:
    Composition:
    Color and texture:
    Do not mention artist, title, exact date, museum, country, movement, or historical period unless it is directly visible as text in the image.

  no_uncertainty: >
    Describe this artwork for an art-history retrieval system.
    Use the following sections exactly:
    Visible content:
    Composition:
    Color and texture:
    Style cues:
    Mention visible objects, composition, color palette, texture, and likely style cues.
    Do not invent the artist name, exact title, exact date, or museum.

  no_style: >
    Describe this artwork for a visual search system.
    Use the following sections exactly:
    Visible content:
    Composition:
    Color and texture:
    Avoid naming art styles, periods, artists, schools, museums, or exact dates.
    Focus on visible objects, layout, colors, brushwork, and image structure.

  compact: >
    Give a compact artwork description in 3 sentences.
    Include visible content, composition, color palette, and only cautious style cues.
    Do not invent artist name, title, date, or museum.

retrieval:
  text_embedding_model: intfloat/e5-small-v2
  clip_model: ViT-B-32
  clip_pretrained: laion2b_s34b_b79k
  batch_text_embed: 128
  batch_clip: 128
  ks: [1, 5, 10]
YAML

cat > src/art_common.py <<'PY'
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
PY

cat > src/build_manifest.py <<'PY'
import argparse
import hashlib
from collections import Counter
from pathlib import Path

import pandas as pd

from art_common import ensure_dir, load_config, normalize_label, set_seed


def image_id_from_path(path: Path, root: Path) -> str:
    rel = str(path.relative_to(root))
    h = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]
    return h


def get_label(path: Path, root: Path, mode: str) -> str:
    rel = path.relative_to(root)
    parts = rel.parts
    if mode == "parent":
        return normalize_label(path.parent.name)
    if mode == "grandparent":
        if len(parts) >= 3:
            return normalize_label(parts[-3])
        return normalize_label(path.parent.name)
    if mode == "top":
        return normalize_label(parts[0])
    raise ValueError(f"Unknown label_mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/art_method.yaml")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--label-mode", default=None, choices=["parent", "grandparent", "top"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["data"].get("seed", 42)))

    root = Path(cfg["paths"]["wikiart_dir"])
    outputs = Path(cfg["paths"]["outputs_dir"])
    ensure_dir(outputs)

    exts = {x.lower() for x in cfg["data"]["image_extensions"]}
    label_mode = args.label_mode or cfg["data"].get("label_mode", "parent")
    max_images = args.max_images if args.max_images is not None else cfg["data"].get("max_images")
    min_per_class = int(cfg["data"].get("min_per_class", 2))

    paths = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    paths = sorted(paths)

    if max_images is not None:
        paths = paths[: int(max_images)]

    rows = []
    for p in paths:
        try:
            label = get_label(p, root, label_mode)
            rows.append(
                {
                    "image_id": image_id_from_path(p, root),
                    "image_path": str(p),
                    "relative_path": str(p.relative_to(root)),
                    "label": label,
                    "label_mode": label_mode,
                }
            )
        except Exception as e:
            print("SKIP", p, e)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found in {root}")

    counts = Counter(df["label"])
    keep = {k for k, v in counts.items() if v >= min_per_class}
    df = df[df["label"].isin(keep)].reset_index(drop=True)

    out = Path(args.out) if args.out else outputs / f"wikiart_manifest_{label_mode}.csv"
    df.to_csv(out, index=False)

    stats = pd.DataFrame(
        [{"label": k, "count": v} for k, v in Counter(df["label"]).most_common()]
    )
    stats_out = outputs / f"wikiart_label_stats_{label_mode}.csv"
    stats.to_csv(stats_out, index=False)

    print(f"saved manifest: {out}")
    print(f"rows          : {len(df)}")
    print(f"classes       : {df['label'].nunique()}")
    print(f"saved stats   : {stats_out}")
    print(stats.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
PY

cat > src/generate_vlm.py <<'PY'
import argparse
import time
from pathlib import Path
from typing import List

import pandas as pd
from PIL import Image
from tqdm import tqdm

from art_common import (
    append_jsonl,
    clean_cuda,
    cuda_oom,
    ensure_dir,
    get_gpu_memory,
    load_config,
    print_gpu,
    read_jsonl,
    set_seed,
)


def load_image(path: str):
    return Image.open(path).convert("RGB")


def load_qwen(model_dir: str):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_dir)
    model.eval()
    return model, processor


def load_llava(model_dir: str):
    import torch
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_dir)
    model.eval()
    return model, processor


def generate_qwen_batch(model, processor, batch: List[dict], prompt: str, max_new_tokens: int, do_sample: bool, temperature: float):
    import torch
    from qwen_vl_utils import process_vision_info

    messages_list = []
    for row in batch:
        messages_list.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": row["image_path"]},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        )

    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]

    image_inputs_all = []
    video_inputs_all = []
    for m in messages_list:
        image_inputs, video_inputs = process_vision_info(m)
        if image_inputs is not None:
            image_inputs_all.extend(image_inputs)
        if video_inputs is not None:
            video_inputs_all.extend(video_inputs)

    inputs = processor(
        text=texts,
        images=image_inputs_all if image_inputs_all else None,
        videos=video_inputs_all if video_inputs_all else None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    outputs = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return outputs


def generate_llava_batch(model, processor, batch: List[dict], prompt: str, max_new_tokens: int, do_sample: bool, temperature: float):
    import torch

    images = [load_image(row["image_path"]) for row in batch]
    conversations = []
    for _ in batch:
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        )

    texts = [
        processor.apply_chat_template(c, add_generation_prompt=True)
        for c in conversations
    ]

    inputs = processor(
        images=images,
        text=texts,
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    outputs = []
    for txt in decoded:
        # Try to remove prompt echo for LLaVA-style outputs.
        if "assistant" in txt.lower():
            parts = txt.split("assistant")
            outputs.append(parts[-1].strip(" :\n\t"))
        else:
            outputs.append(txt.strip())
    return outputs


def initial_batch(kind: str, cfg: dict) -> int:
    gen = cfg["generation"]
    used, total, util = get_gpu_memory()
    if kind == "qwen25vl":
        base = int(gen.get("start_batch_qwen", 8))
        max_bs = int(gen.get("max_batch_qwen", 32))
    else:
        base = int(gen.get("start_batch_llava", 16))
        max_bs = int(gen.get("max_batch_llava", 64))

    if total is None:
        return base

    free = total - used
    if free > 70:
        return min(max_bs, base * 4)
    if free > 40:
        return min(max_bs, base * 2)
    if free > 20:
        return base
    return max(1, base // 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/art_method.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--ablation", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["data"].get("seed", 42)))

    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    ensure_dir(outputs_dir)

    manifest = Path(args.manifest) if args.manifest else outputs_dir / f"wikiart_manifest_{cfg['data'].get('label_mode', 'parent')}.csv"
    df = pd.read_csv(manifest)
    if args.limit:
        df = df.head(args.limit).copy()

    model_cfg = cfg["models"][args.model_key]
    kind = model_cfg["kind"]
    model_dir = model_cfg["local_dir"]
    prompt = cfg["ablations"][args.ablation]

    out_path = outputs_dir / "generations" / f"{args.model_key}__{args.ablation}.jsonl"
    ensure_dir(out_path.parent)

    done = set()
    if args.resume and out_path.exists():
        for r in read_jsonl(out_path):
            done.add(r["image_id"])

    rows = df.to_dict("records")
    rows = [r for r in rows if r["image_id"] not in done]

    print(f"model_key : {args.model_key}")
    print(f"kind      : {kind}")
    print(f"ablation  : {args.ablation}")
    print(f"manifest  : {manifest}")
    print(f"remaining : {len(rows)}")
    print(f"out       : {out_path}")
    print_gpu("before load")

    if kind == "qwen25vl":
        model, processor = load_qwen(model_dir)
        gen_fn = generate_qwen_batch
    elif kind == "llava_onevision":
        model, processor = load_llava(model_dir)
        gen_fn = generate_llava_batch
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    print_gpu("after load")

    gen_cfg = cfg["generation"]
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 220))
    do_sample = bool(gen_cfg.get("do_sample", False))
    temperature = float(gen_cfg.get("temperature", 0.0))
    increase_every = int(gen_cfg.get("increase_every_success_batches", 8))

    if args.batch_size and args.batch_size > 0:
        bs = args.batch_size
    else:
        bs = initial_batch(kind, cfg)

    max_bs = int(gen_cfg.get("max_batch_qwen" if kind == "qwen25vl" else "max_batch_llava", 32))

    i = 0
    success_batches = 0
    pbar = tqdm(total=len(rows), dynamic_ncols=True)

    while i < len(rows):
        batch = rows[i : i + bs]
        t0 = time.time()

        try:
            texts = gen_fn(
                model,
                processor,
                batch,
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
            out_rows = []
            for row, txt in zip(batch, texts):
                out_rows.append(
                    {
                        "image_id": row["image_id"],
                        "image_path": row["image_path"],
                        "relative_path": row.get("relative_path", ""),
                        "label": row.get("label", ""),
                        "model_key": args.model_key,
                        "model_kind": kind,
                        "ablation": args.ablation,
                        "prompt": prompt,
                        "response": txt,
                        "max_new_tokens": max_new_tokens,
                        "do_sample": do_sample,
                        "temperature": temperature,
                        "batch_size_used": bs,
                        "seconds": time.time() - t0,
                    }
                )
            append_jsonl(out_path, out_rows)
            i += len(batch)
            pbar.update(len(batch))
            success_batches += 1

            if success_batches >= increase_every and bs < max_bs:
                used, total, util = get_gpu_memory()
                if util is None or util < float(gen_cfg.get("target_gpu_mem_util", 0.92)):
                    new_bs = min(max_bs, bs * 2)
                    if new_bs != bs:
                        print(f"\nIncreasing batch size: {bs} -> {new_bs}")
                        bs = new_bs
                success_batches = 0

        except Exception as e:
            if cuda_oom(e):
                print(f"\nCUDA OOM at batch size {bs}. Reducing batch.")
                clean_cuda()
                success_batches = 0
                if bs <= 1:
                    print("OOM even at batch size 1. Skipping one problematic image.")
                    bad = rows[i]
                    append_jsonl(
                        out_path,
                        [
                            {
                                "image_id": bad["image_id"],
                                "image_path": bad["image_path"],
                                "relative_path": bad.get("relative_path", ""),
                                "label": bad.get("label", ""),
                                "model_key": args.model_key,
                                "model_kind": kind,
                                "ablation": args.ablation,
                                "prompt": prompt,
                                "response": "",
                                "error": "CUDA OOM at batch size 1",
                                "batch_size_used": bs,
                            }
                        ],
                    )
                    i += 1
                    pbar.update(1)
                else:
                    bs = max(1, bs // 2)
                    print(f"New batch size: {bs}")
            else:
                clean_cuda()
                print(f"\nERROR at index {i}, batch size {bs}: {repr(e)}")
                if bs > 1:
                    bs = max(1, bs // 2)
                    print(f"Reducing batch size to {bs} and retrying.")
                else:
                    bad = rows[i]
                    append_jsonl(
                        out_path,
                        [
                            {
                                "image_id": bad["image_id"],
                                "image_path": bad["image_path"],
                                "relative_path": bad.get("relative_path", ""),
                                "label": bad.get("label", ""),
                                "model_key": args.model_key,
                                "model_kind": kind,
                                "ablation": args.ablation,
                                "prompt": prompt,
                                "response": "",
                                "error": repr(e),
                                "batch_size_used": bs,
                            }
                        ],
                    )
                    i += 1
                    pbar.update(1)

    pbar.close()
    print("\nDONE")
    print(out_path)
    print_gpu("final")


if __name__ == "__main__":
    main()
PY

cat > src/build_baselines.py <<'PY'
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from art_common import ensure_dir, load_config, set_seed, write_json


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def save_npz(path: Path, ids, labels, embeddings):
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        image_ids=np.array(ids),
        labels=np.array(labels),
        embeddings=embeddings.astype("float32"),
    )
    print("saved:", path, embeddings.shape)


def build_clip(manifest: Path, cfg: dict):
    import open_clip

    df = pd.read_csv(manifest)
    out = Path(cfg["paths"]["outputs_dir"]) / "embeddings" / "baseline_clip_image.npz"

    model_name = cfg["retrieval"]["clip_model"]
    pretrained = cfg["retrieval"]["clip_pretrained"]
    batch_size = int(cfg["retrieval"].get("batch_clip", 128))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()

    all_embs = []
    ids = []
    labels = []

    rows = df.to_dict("records")
    i = 0
    bs = batch_size

    pbar = tqdm(total=len(rows), dynamic_ncols=True, desc="CLIP image baseline")
    while i < len(rows):
        batch = rows[i : i + bs]
        try:
            imgs = []
            good = []
            for r in batch:
                try:
                    im = Image.open(r["image_path"]).convert("RGB")
                    imgs.append(preprocess(im))
                    good.append(r)
                except Exception:
                    pass

            if not imgs:
                i += len(batch)
                pbar.update(len(batch))
                continue

            tensor = torch.stack(imgs).to(device)
            with torch.no_grad():
                emb = model.encode_image(tensor)
                emb = emb.float()
                emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embs.append(emb.cpu().numpy())
            ids.extend([r["image_id"] for r in good])
            labels.extend([r["label"] for r in good])
            i += len(batch)
            pbar.update(len(batch))

        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs > 1:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                print("CLIP OOM, new batch:", bs)
            else:
                raise

    pbar.close()
    embs = np.concatenate(all_embs, axis=0)
    save_npz(out, ids, labels, embs)


def build_metadata_text(manifest: Path, cfg: dict):
    from sentence_transformers import SentenceTransformer

    df = pd.read_csv(manifest)
    out = Path(cfg["paths"]["outputs_dir"]) / "embeddings" / "baseline_metadata_label_text.npz"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = cfg["retrieval"]["text_embedding_model"]
    model = SentenceTransformer(model_name, device=device)

    texts = [
        f"artwork style or category: {label}"
        for label in df["label"].astype(str).tolist()
    ]

    embs = model.encode(
        texts,
        batch_size=int(cfg["retrieval"].get("batch_text_embed", 128)),
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    save_npz(
        out,
        df["image_id"].astype(str).tolist(),
        df["label"].astype(str).tolist(),
        np.asarray(embs),
    )


def build_random(manifest: Path, cfg: dict):
    df = pd.read_csv(manifest)
    rng = np.random.default_rng(int(cfg["data"].get("seed", 42)))
    embs = rng.normal(size=(len(df), 512)).astype("float32")
    embs = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
    out = Path(cfg["paths"]["outputs_dir"]) / "embeddings" / "baseline_random.npz"
    save_npz(
        out,
        df["image_id"].astype(str).tolist(),
        df["label"].astype(str).tolist(),
        embs,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/art_method.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--which", nargs="+", default=["clip", "metadata", "random"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["data"].get("seed", 42)))

    manifest = Path(args.manifest) if args.manifest else Path(cfg["paths"]["outputs_dir"]) / f"wikiart_manifest_{cfg['data'].get('label_mode', 'parent')}.csv"

    if "clip" in args.which:
        build_clip(manifest, cfg)
    if "metadata" in args.which:
        build_metadata_text(manifest, cfg)
    if "random" in args.which:
        build_random(manifest, cfg)

    write_json(Path(cfg["paths"]["outputs_dir"]) / "baselines_done.json", {"done": args.which})


if __name__ == "__main__":
    main()
PY

cat > src/embed_generations.py <<'PY'
import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from art_common import ensure_dir, load_config, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/art_method.yaml")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    rows = read_jsonl(args.input_jsonl)
    rows = [r for r in rows if r.get("response", "").strip()]
    if not rows:
        raise RuntimeError(f"No valid rows in {args.input_jsonl}")

    texts = []
    ids = []
    labels = []

    for r in rows:
        txt = r.get("response", "")
        texts.append("query: " + txt)
        ids.append(r["image_id"])
        labels.append(r.get("label", ""))

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = SentenceTransformer(cfg["retrieval"]["text_embedding_model"], device=device)

    embs = model.encode(
        texts,
        batch_size=int(cfg["retrieval"].get("batch_text_embed", 128)),
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    in_path = Path(args.input_jsonl)
    if args.out:
        out = Path(args.out)
    else:
        out = Path(cfg["paths"]["outputs_dir"]) / "embeddings" / f"{in_path.stem}.npz"

    ensure_dir(out.parent)
    np.savez_compressed(
        out,
        image_ids=np.array(ids),
        labels=np.array(labels),
        embeddings=np.asarray(embs).astype("float32"),
    )
    print("saved:", out, np.asarray(embs).shape)


if __name__ == "__main__":
    main()
PY

cat > src/evaluate.py <<'PY'
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
PY

cat > src/run_full_pipeline.py <<'PY'
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from art_common import ensure_dir, load_config


def run(cmd, log_path=None):
    print("\n" + "=" * 100)
    print("CMD:", " ".join(cmd))
    print("=" * 100)

    if log_path:
        ensure_dir(Path(log_path).parent)
        with open(log_path, "a", encoding="utf-8") as f:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
            ret = p.wait()
    else:
        ret = subprocess.call(cmd)

    if ret != 0:
        raise RuntimeError(f"Command failed with code {ret}: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/art_method.yaml")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=["qwen25vl_3b", "llava_onevision_0_5b"])
    parser.add_argument("--ablations", nargs="+", default=["full", "visual_only", "no_uncertainty", "no_style", "compact"])
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-to-files", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    outputs = Path(cfg["paths"]["outputs_dir"])
    logs = Path(cfg["paths"]["project_dir"]) / "logs"

    manifest = outputs / f"wikiart_manifest_{cfg['data'].get('label_mode', 'parent')}.csv"

    cmd = [
        sys.executable,
        "src/build_manifest.py",
        "--config",
        args.config,
        "--out",
        str(manifest),
    ]
    if args.max_images:
        cmd += ["--max-images", str(args.max_images)]
    run(cmd, logs / "build_manifest.log" if args.log_to_files else None)

    if not args.skip_generation:
        for model_key in args.models:
            for ablation in args.ablations:
                log = logs / f"generate__{model_key}__{ablation}.log" if args.log_to_files else None
                cmd = [
                    sys.executable,
                    "src/generate_vlm.py",
                    "--config",
                    args.config,
                    "--manifest",
                    str(manifest),
                    "--model-key",
                    model_key,
                    "--ablation",
                    ablation,
                    "--resume",
                ]
                run(cmd, log)

    if not args.skip_baselines:
        run(
            [
                sys.executable,
                "src/build_baselines.py",
                "--config",
                args.config,
                "--manifest",
                str(manifest),
            ],
            logs / "build_baselines.log" if args.log_to_files else None,
        )

    gen_dir = outputs / "generations"
    for js in sorted(gen_dir.glob("*.jsonl")):
        out_npz = outputs / "embeddings" / f"{js.stem}.npz"
        if out_npz.exists():
            print("skip existing embedding:", out_npz)
            continue
        run(
            [
                sys.executable,
                "src/embed_generations.py",
                "--config",
                args.config,
                "--input-jsonl",
                str(js),
                "--out",
                str(out_npz),
            ],
            logs / f"embed__{js.stem}.log" if args.log_to_files else None,
        )

    run(
        [sys.executable, "src/evaluate.py", "--config", args.config],
        logs / "evaluate.log" if args.log_to_files else None,
    )

    print("\nDONE FULL PIPELINE")
    print("Main result table:")
    print(outputs / "summary_metrics.csv")


if __name__ == "__main__":
    main()
PY

cat > scripts/run_debug_200.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd /home/tahiti/ARTeccv
source .venv/bin/activate

python src/run_full_pipeline.py \
  --config configs/art_method.yaml \
  --max-images 200 \
  --resume
SH

cat > scripts/run_full_all.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd /home/tahiti/ARTeccv
source .venv/bin/activate

python src/run_full_pipeline.py \
  --config configs/art_method.yaml \
  --resume \
  --log-to-files
SH

cat > scripts/run_only_qwen_full.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd /home/tahiti/ARTeccv
source .venv/bin/activate

python src/run_full_pipeline.py \
  --config configs/art_method.yaml \
  --models qwen25vl_3b \
  --ablations full \
  --resume
SH

cat > scripts/run_only_llava_full.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd /home/tahiti/ARTeccv
source .venv/bin/activate

python src/run_full_pipeline.py \
  --config configs/art_method.yaml \
  --models llava_onevision_0_5b \
  --ablations full \
  --resume
SH

cat > scripts/watch_gpu.sh <<'SH'
#!/usr/bin/env bash
watch -n 1 nvidia-smi
SH

chmod +x scripts/*.sh

python - <<'PY'
import torch, numpy, transformers
print("torch:", torch.__version__)
print("cuda :", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO")
print("numpy:", numpy.__version__)
print("transformers:", transformers.__version__)
PY

echo ""
echo "Installed full ARTeccv method pipeline."
echo ""
echo "Debug run:"
echo "  bash scripts/run_debug_200.sh"
echo ""
echo "Full run:"
echo "  bash scripts/run_full_all.sh"
