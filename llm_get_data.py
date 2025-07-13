import json
from datetime import datetime
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pytesseract
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import requests  # for calling local LLM API

CHARACTER_DB_PATH = Path("../characters.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load emotion model
EMOTION_MODEL_NAME = "nateraw/bert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME)
emotion_model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_NAME).to(DEVICE)
EMOTIONS = emotion_model.config.id2label

# ---------------------------
# Load/Save Character DB
# ---------------------------
def load_character_db():
    if CHARACTER_DB_PATH.exists():
        with open(CHARACTER_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_character_db(db):
    with open(CHARACTER_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

# ---------------------------
# Emotion Detection
# ---------------------------
def detect_emotion(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(DEVICE)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
        probs = F.softmax(logits, dim=1)
    top_id = probs.argmax().item()
    return EMOTIONS[top_id], round(probs[0][top_id].item(), 3)

# ---------------------------
# Text Context Classification
# ---------------------------
def classify_text_context(text: str, character_db: dict) -> dict:
    is_thought = "..." in text or "I wonder" in text
    is_action = any(w in text.lower() for w in ["*crash*", "boom", "slam", "whoosh", "slashes", "flies"])
    is_speech = not is_thought and not is_action

    perspective = "reader"
    if any(pronoun in text for pronoun in ["I", "me", "my"]):
        perspective = "character"

    speaker_name = "unknown"
    if "arthur" in text.lower():
        speaker_name = "Arthur"

    emotion, confidence = detect_emotion(text)

    speaker_profile = character_db.get(speaker_name, {
        "name": speaker_name,
        "traits": [],
        "role": "unknown",
        "first_seen": datetime.now().isoformat()
    })

    return {
        "raw_text": text,
        "type": "thought" if is_thought else "action" if is_action else "speech",
        "tone": "internal" if is_thought else "out loud" if is_speech else "effect",
        "emotion": emotion,
        "confidence": confidence,
        "perspective": perspective,
        "speaker": speaker_profile
    }

# ---------------------------
# New: Clean text via local LLM (Ollama API)
# ---------------------------
def clean_text_via_llm(raw_text: str) -> str:
    prompt = f"""
        You are a manga dialogue cleaner.

        Your job is to fix raw OCR text from manga so that it reads clearly, logically, and is ready to be spoken aloud. Correct punctuation, spelling, and grammar. Preserve the original meaning, manga tone, and emotional beats.

        - Keep it in natural spoken English (fantasy tone is okay).
        - DO NOT add new characters, names, or explanations.
        - Feel free to slightly reword awkward or broken lines so they make sense.
        - Break long thoughts into short lines for dramatic effect.
        - Output should sound like authentic manga dialogue.
        - Return only the cleaned dialogue. No quotes, no speaker tags.

        OCR TEXT:
        {raw_text}

        CLEANED DIALOGUE:
        """.strip()
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            },
            timeout=10
        )
        response.raise_for_status()
        result_json = response.json()
        return result_json.get("response", "").strip() or raw_text.strip()
    except Exception as e:
        print(f"⚠️ LLM cleaning failed: {e}")
        return raw_text.strip()

# ---------------------------
# OCR + LLM Image Processor
# ---------------------------
def get_image_data(image_path: Path, character_db: dict) -> dict:
    image = Image.open(image_path)
    image = image.convert("RGB") if image.mode != "RGB" else image

    # Use OCR instead of hallucination-prone VLM
    extracted_text = pytesseract.image_to_string(image).strip()

    # Call LLM to clean the OCR text
    cleaned_text = clean_text_via_llm(extracted_text)

    analysis = classify_text_context(extracted_text, character_db)

    # Update character DB if speaker is new
    speaker_name = analysis["speaker"]["name"]
    if speaker_name != "unknown" and speaker_name not in character_db:
        character_db[speaker_name] = analysis["speaker"]

    return {
        "image": str(image_path),
        "timestamp": datetime.now().isoformat(),
        "extracted_text": extracted_text,
        "cleaned_text": cleaned_text,              # <== added cleaned text here
        "analysis": analysis
    }

# ---------------------------
# Process Folder of Images
# ---------------------------
def run_on_folder(folder_path: str) -> dict:
    folder = Path(folder_path)
    image_paths = sorted(folder.glob("*.jpg"))
    if not image_paths:
        print("❌ No images found.")
        return {}

    chapter_results = []
    character_db = load_character_db()

    print(f"📚 Processing Chapter: {folder.name}")
    for img_path in tqdm(image_paths, desc="🖼️ Analyzing"):
        data = get_image_data(img_path, character_db)
        chapter_results.append(data)

    # Save results
    out_path = folder / f"chapter_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chapter_results, f, indent=2, ensure_ascii=False)

    save_character_db(character_db)

    return {
        "chapter": folder.name,
        "images_processed": len(image_paths),
        "character_count": len(character_db),
        "output": str(out_path)
    }