import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_DIR / "models"
HF_CACHE = PROJECT_DIR / "hf_cache"

DEFAULT_MODELS = {
    "qwen2_5_vl_3b_instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
    "llava_onevision_qwen2_0_5b_ov_hf": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
}

LARGE_MODELS = {
    "qwen2_5_vl_7b_instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "llava_onevision_qwen2_7b_ov_hf": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
}


def download_model(local_name: str, repo_id: str) -> None:
    target_dir = MODELS_DIR / local_name
    target_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print(f"Downloading repo : {repo_id}")
    print(f"Saving to        : {target_dir}")
    print("=" * 90)

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        cache_dir=str(HF_CACHE),
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=[
            "*.msgpack",
            "*.h5",
            "*.ot",
            "*.onnx",
            "*.tflite",
            "flax_model*",
            "tf_model*",
        ],
    )

    print(f"OK: {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--large",
        action="store_true",
        help="Also download 7B versions of Qwen2.5-VL and LLaVA-OneVision.",
    )
    args = parser.parse_args()

    os.environ["HF_HOME"] = str(HF_CACHE)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)

    models = dict(DEFAULT_MODELS)

    if args.large:
        models.update(LARGE_MODELS)

    print("Models selected:")
    for local_name, repo_id in models.items():
        print(f"  {local_name:40s} <- {repo_id}")

    for local_name, repo_id in models.items():
        download_model(local_name, repo_id)

    print("\nDONE.")
    print(f"Models directory: {MODELS_DIR}")


if __name__ == "__main__":
    main()
