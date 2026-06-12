from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models" / "qwen2_5_vl_3b_instruct"
WIKIART_DIR = PROJECT_DIR / "data" / "wikiart"


def find_first_image(root: Path) -> Path:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            return p
    raise FileNotFoundError(f"No image found in {root}")


image_path = find_first_image(WIKIART_DIR)
print(f"Image: {image_path}")
print(f"Model: {MODEL_DIR}")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype="auto",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_DIR)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
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

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
    )

generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print("\nQWEN OUTPUT:")
print(output_text[0])
