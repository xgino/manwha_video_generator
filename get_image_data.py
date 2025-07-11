import os
import re
import json
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from transformers import pipeline
from ultralytics import YOLO
import torch

# Configuration
MODEL_PATH = "yolov8l.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥️  Using Device:", DEVICE)
CONFIDENCE_THRESHOLD = 0.5
TEXT_PADDING = 4  # padding around text boxes
OCR_CONFIDENCE_THRESHOLD = 60

# Load models
yolo_model = YOLO(MODEL_PATH)

def enhance_contrast(image, factor=2.0):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def extract_text_boxes(image):
    """Extract OCR text bounding boxes from the image using pytesseract."""
    # Use RGB directly and optionally enhance contrast
    enhanced = enhance_contrast(image)
    
    # Use PSM 6 (assumes block of text); you can experiment with other PSMs
    custom_config = r"--psm 11"
    data = pytesseract.image_to_data(enhanced, lang="eng", config=custom_config, output_type=pytesseract.Output.DICT)
    
    boxes = []
    n = len(data['text'])
    for i in range(n):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        if not text or conf < OCR_CONFIDENCE_THRESHOLD:
            continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        boxes.append({
            "bbox": [x, y, w, h],
            "text": text,
            "confidence": conf
        })
    return boxes

def analyze_text(text):
    uppercase_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
    exclamations = text.count("!")
    return {
        "text": text,
        "text_style": {
            "uppercase_ratio": round(uppercase_ratio, 2),
            "exclamation_count": exclamations
        }
    }

def crop_box_with_padding(image, bbox, padding=TEXT_PADDING):
    x, y, w, h = bbox
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.width, x + w + padding)
    y2 = min(image.height, y + h + padding)
    return image.crop((x1, y1, x2, y2))

def process_image(image_path, image_name):
    image = Image.open(image_path).convert("RGB")
    text_boxes = extract_text_boxes(image)

    results = []
    for idx, box in enumerate(text_boxes):
        text = box["text"]
        if not text:
            continue
        analysis = analyze_text(text)
        results.append({
            "type": "text",
            "image": image_name,
            "box_id": idx + 1,
            "bbox": box["bbox"],
            "confidence": box["confidence"],
            "text": analysis["text"],
            "text_style": analysis["text_style"]
        })
    return results

def numeric_sort_key(path):
    return int(re.match(r"\d+", path.stem).group()) if re.match(r"\d+", path.stem) else path.stem

def run_on_folder(folder_path):
    image_files = sorted(Path(folder_path).glob("*.jpg"), key=numeric_sort_key)
    result = {}

    for img_file in tqdm(image_files, desc="Get data from image"):
        data = process_image(img_file, img_file.name)
        result[img_file.name] = data

    output_path = Path(folder_path) / "text_boxes_by_image.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)