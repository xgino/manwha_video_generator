import os
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ---------------------------
# Configuration
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Salesforce/blip2-opt-2.7b"  # or try: "Salesforce/blip2-flan-t5-xl"

print("🧠 Loading BLIP-2 model on", DEVICE)
processor = Blip2Processor.from_pretrained(MODEL_NAME)
model = Blip2ForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32)
model.to(DEVICE)

# ---------------------------
# Inference Function
# ---------------------------
def get_image_text(image_path: Path) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE, torch.float16 if DEVICE == "cuda" else torch.float32)
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

# ---------------------------
# Process Folder of Images
# ---------------------------
def run_on_folder(folder_path: str) -> dict:
    image_paths = sorted(Path(folder_path).glob("*.jpg"))
    results = {}

    for image_path in tqdm(image_paths, desc="🔍 Processing Images"):
        try:
            text = get_image_text(image_path)
            results[image_path.name] = text
        except Exception as e:
            print(f"❌ Failed on {image_path.name}: {e}")
            results[image_path.name] = None

    # Save to JSON
    out_path = Path(folder_path) / "llm_text_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results

