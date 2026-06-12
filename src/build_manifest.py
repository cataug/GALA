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
