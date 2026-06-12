from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models" / "llava_onevision_qwen2_0_5b_ov_hf"
WIKIART_DIR = PROJECT_DIR / "data" / "wikiart"


def find_first_image(root: Path) -> Path:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            return p
    raise FileNotFoundError(f"No image found in {root}")


image_path = find_first_image(WIKIART_DIR)
image = Image.open(image_path).convert("RGB")

print(f"Image: {image_path}")
print(f"Model: {MODEL_DIR}")

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_DIR)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": (
                    "Describe this artwork for an art-history retrieval system. "
                    "Mention visible style, objects, composition, color palette, and possible period cues. "
                    "Do not invent the artist name."
                ),
            },
        ],
    }
]

prompt = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
)

inputs = processor(
    images=image,
    text=prompt,
    return_tensors="pt",
)

inputs = {
    k: v.to(model.device) if hasattr(v, "to") else v
    for k, v in inputs.items()
}

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
    )

output = processor.decode(
    generated_ids[0],
    skip_special_tokens=True,
)

print("\nLLAVA-ONEVISION OUTPUT:")
print(output)
