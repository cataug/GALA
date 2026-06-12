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
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None and processor.tokenizer.eos_token is not None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
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
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None and processor.tokenizer.eos_token is not None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
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
