from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
WIKIART_DIR = PROJECT_DIR / "data" / "wikiart"
MODELS_DIR = PROJECT_DIR / "models"

print("PROJECT:", PROJECT_DIR)

required = [
    WIKIART_DIR,
    MODELS_DIR,
    PROJECT_DIR / "hf_cache",
    PROJECT_DIR / "scripts",
    PROJECT_DIR / "src",
    PROJECT_DIR / "configs",
    PROJECT_DIR / "outputs",
    PROJECT_DIR / "logs",
]

for p in required:
    print(f"{'OK  ' if p.exists() else 'MISS'} {p}")

exts = [".jpg", ".jpeg", ".png", ".webp"]
images = [p for p in WIKIART_DIR.rglob("*") if p.suffix.lower() in exts]

print(f"\nWikiArt images found: {len(images)}")

for p in images[:10]:
    print(" ", p)

print("\nDownloaded model dirs:")
for p in sorted(MODELS_DIR.glob("*")):
    if p.is_dir():
        print(" ", p.name)
