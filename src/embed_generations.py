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
