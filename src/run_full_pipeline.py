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
