import json
import re
from tqdm import tqdm
import unicodedata
import string
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration

import nltk
from nltk.corpus import words as nltk_words
nltk.download('words')
ENGLISH_WORDS = set(nltk_words.words())

import torch
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")  # Apple Metal GPU backend
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("🖥️  Using Device:", DEVICE)


# Load once
tokenizer = T5Tokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws", use_fast=False)
model = T5ForConditionalGeneration.from_pretrained("Vamsi/T5_Paraphrase_Paws")
denoise_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=DEVICE)

# Optional: your emotion classifier here
emotion_pipeline = lambda x: [{"label": "neutral"}]  # Stub if none loaded

# Stub emotion classifier (can replace)
emotion_pipeline = lambda x: [{"label": "neutral"}]

def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    text = re.sub(r"[^\w\s.!?,'-]", "", text)  # keep basic punctuation only
    text = re.sub(r"\s+", " ", text)
    return text

def deduplicate_consecutive_tokens(text, max_repeats=2):
    tokens = text.split()
    cleaned = []
    prev = None
    count = 0

    for tok in tokens:
        if tok.lower() == prev:
            count += 1
        else:
            count = 1
            prev = tok.lower()

        if count <= max_repeats:
            cleaned.append(tok)
    return " ".join(cleaned)

def contains_real_words(text, threshold=0.4):
    tokens = text.lower().split()
    if not tokens:
        return False
    real = sum(1 for t in tokens if t in ENGLISH_WORDS)
    return (real / len(tokens)) >= threshold

def clean_and_combine_text_group(group):
    raw_texts = [normalize_text(i["text"]) for i in group if i.get("text")]
    combined = " ".join(raw_texts)
    combined = deduplicate_consecutive_tokens(combined)

    # Short, clean lines should skip paraphrasing
    skip_denoise = len(combined.split()) <= 8 and contains_real_words(combined)

    if not skip_denoise:
        try:
            out = denoise_pipeline(
                f"paraphrase: {combined}",
                max_length=64,
                do_sample=True,
                top_p=0.92,
                temperature=0.8
            )[0]["generated_text"].strip()
        except Exception:
            out = combined.strip()
    else:
        out = combined.strip()

    # Final checks
    if not out or len(out) < 3:
        return ""
    if not contains_real_words(out, threshold=0.3):  # relax for short but real lines
        return ""

    return out

def clean_bubble_data(folder_path):
    json_path = Path(folder_path) / "text_boxes_by_image.json"
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    final_output = {}

    for image_name, items in tqdm(raw_data.items(), desc="Processing"):
        # Sort fragments top-to-bottom by bbox y if available, else keep order
        sorted_items = sorted(items, key=lambda i: i.get("bbox", [0, 9999])[1])

        # Merge all fragments into one group (no splitting by type)
        cleaned_text = clean_and_combine_text_group(sorted_items)
        if not cleaned_text:
            # No valid text for this image
            final_output[image_name] = {"image": image_name, "text": ""}
            continue

        # Optionally add emotion detection here if needed (for full text)
        emotion = None
        try:
            emotion = emotion_pipeline(cleaned_text)[0]["label"].lower()
        except Exception:
            pass

        entry = {"image": image_name, "text": cleaned_text}
        if emotion:
            entry["emotion"] = emotion

        final_output[image_name] = entry

    return final_output