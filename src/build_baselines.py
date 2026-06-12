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
