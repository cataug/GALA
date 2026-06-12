# GALA: Grounded Art Language Analysis

**GALA** is a vision-language evaluation pipeline for artwork understanding, retrieval, metadata alignment, and unsupported-claim analysis.

The project studies whether open vision-language models can generate artwork descriptions that are useful not only as captions, but also as structured representations for cultural search and cautious art-historical analysis.

GALA currently supports experiments with:

- **Qwen2.5-VL**
- **LLaVA-OneVision**
- **WikiArt**
- **CLIP image retrieval baseline**
- **metadata-text retrieval baseline**
- **random retrieval baseline**

---

## Overview

Large digitized art collections often contain incomplete, heterogeneous, or coarse metadata. Vision-language models can describe artworks from images, but they may also introduce unsupported claims about artists, periods, titles, styles, iconography, or cultural origin.

GALA treats artwork description as a **grounded representation problem**:

> Can generated vision-language descriptions support artwork retrieval while separating visible evidence, metadata-supported information, and uncertain interpretation?

The pipeline evaluates generated descriptions through:

- retrieval quality;
- metadata alignment;
- prompt ablations;
- unsupported-claim proxy analysis.

---

## Pipeline

GALA follows four main stages:

1. **Artwork preprocessing**  
   WikiArt images are indexed, filtered, converted to RGB, and stored in a manifest.

2. **Vision-language generation**  
   Qwen2.5-VL and LLaVA-OneVision generate artwork descriptions under matched prompt settings.

3. **Structured representation**  
   Generated responses are parsed into fields such as visible content, composition, color and texture, style cues, and uncertain hypotheses.

4. **Evaluation**  
   Descriptions are embedded and evaluated using retrieval metrics, metadata alignment, and unsupported-claim analysis.

---

## Repository Structure

```text
GALA/
├── configs/
│   └── art_method.yaml
├── scripts/
│   ├── download_models.py
│   ├── test_qwen25vl.py
│   ├── test_llava_onevision.py
│   ├── run_debug_200.sh
│   ├── run_full_all.sh
│   └── watch_gpu.sh
├── src/
│   ├── art_common.py
│   ├── build_manifest.py
│   ├── generate_vlm.py
│   ├── build_baselines.py
│   ├── embed_generations.py
│   ├── evaluate.py
│   └── run_full_pipeline.py
├── outputs/
│   ├── generations/
│   ├── embeddings/
│   └── summary_metrics.csv
├── logs/
└── requirements.txt
```

Model weights and raw datasets are not included.

---

## Installation

```bash
git clone git@github.com:cataug/GALA.git
cd GALA

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

For GPU inference, install a CUDA-compatible PyTorch build, for example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Dataset

GALA expects WikiArt to be available locally.

Default path:

```text
data/wikiart
```

If WikiArt is stored elsewhere, create a symbolic link:

```bash
mkdir -p data
ln -s /path/to/wikiart data/wikiart
```

Example:

```bash
ln -s /home/tahiti/UME2026/data/wikiart data/wikiart
```

---

## Models

Download the default models:

```bash
python scripts/download_models.py
```

Default models:

```text
Qwen/Qwen2.5-VL-3B-Instruct
llava-hf/llava-onevision-qwen2-0.5b-ov-hf
```

Optional 7B models:

```bash
python scripts/download_models.py --large
```

Models are stored in:

```text
models/
```

This directory is ignored by Git.

---

## Quick Tests

Test Qwen2.5-VL:

```bash
python scripts/test_qwen25vl.py
```

Test LLaVA-OneVision:

```bash
python scripts/test_llava_onevision.py
```

---

## Running Experiments

### Debug run

Run the full pipeline on a small subset:

```bash
bash scripts/run_debug_200.sh
```

This checks manifest construction, model generation, embeddings, baselines, and evaluation.

### Full run

Run the full experiment:

```bash
bash scripts/run_full_all.sh
```

The full experiment evaluates:

- Qwen2.5-VL;
- LLaVA-OneVision;
- full prompt;
- visual-only prompt;
- no-uncertainty prompt;
- no-style prompt;
- compact prompt;
- CLIP image retrieval baseline;
- metadata-text retrieval baseline;
- random baseline.

---

## Prompt Variants

| Variant | Description |
|---|---|
| `full` | Full artwork description with visible evidence, style cues, and uncertain hypotheses |
| `visual_only` | Restricts the model to directly visible content |
| `no_uncertainty` | Removes explicit uncertainty separation |
| `no_style` | Avoids style and period naming |
| `compact` | Produces a short artwork description |

---

## Evaluation Metrics

The main result table is saved to:

```text
outputs/summary_metrics.csv
```

| Metric | Meaning |
|---|---|
| `recall@1` | Retrieval hit at rank 1 |
| `recall@5` | Retrieval hit within top 5 |
| `recall@10` | Retrieval hit within top 10 |
| `map@K` | Mean average precision at K |
| `lexical_label_alignment` | Lexical agreement with metadata labels |
| `hallucination_proxy_rate` | Automatic unsupported-claim proxy |
| `avg_words` | Average response length |

The unsupported-claim score is an automatic proxy. It is not a substitute for expert factuality annotation.

---

## Baselines

GALA compares VLM-generated descriptions against:

| Baseline | Description |
|---|---|
| CLIP image retrieval | Retrieval using image embeddings |
| Metadata-text retrieval | Retrieval using textual metadata labels |
| Random retrieval | Sanity-check baseline |

---

## Monitoring

Check generation progress:

```bash
python - <<'PY'
import glob, time
from pathlib import Path

for f in sorted(glob.glob("outputs/generations/*.jsonl")):
    p = Path(f)
    lines = sum(1 for _ in open(p, "r", encoding="utf-8"))
    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
    print(f"{p.name:45s} rows={lines:6d} modified={mtime}")
PY
```

Watch GPU usage:

```bash
watch -n 1 nvidia-smi
```

---

## Example Outputs

```text
outputs/
├── wikiart_manifest_parent.csv
├── wikiart_label_stats_parent.csv
├── generations/
│   ├── qwen25vl_3b__full.jsonl
│   ├── qwen25vl_3b__visual_only.jsonl
│   ├── qwen25vl_3b__no_uncertainty.jsonl
│   ├── llava_onevision_0_5b__full.jsonl
│   └── ...
├── embeddings/
│   ├── qwen25vl_3b__full.npz
│   ├── baseline_clip_image.npz
│   ├── baseline_metadata_label_text.npz
│   └── baseline_random.npz
└── summary_metrics.csv
```

---

## Paper Context

This repository supports the project:

**GALA: Grounded Vision-Language Models for Artwork Retrieval and Cultural Analysis**

The study evaluates whether open vision-language models can produce structured artwork descriptions that support semantic retrieval while tracking unsupported art-historical claims.

---

## Citation

```bibtex
@misc{gala2026,
  title        = {GALA: Grounded Art Language Analysis for Artwork Retrieval and Cultural Analysis},
  author       = {??},
  year         = {2026},
}
```

---

## Notes

- Model weights are not included.
- Raw WikiArt images are not included.
- Generated outputs and logs may be included for reproducibility.
- The unsupported-claim metric is an automatic screening signal.
- For art-historical use, generated claims should be verified against collection metadata or expert annotation.

---

## License

Add a license before public reuse.
