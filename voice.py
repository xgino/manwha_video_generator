import json
import os
import re
import pyttsx3
from pathlib import Path


def clean_text(text: str) -> str:
    """Clean and format text to improve TTS output."""
    text = text.replace("\n", " ").strip()
    text = re.sub(r"--+", ",", text)              # Replace dashes with commas
    text = re.sub(r"\.\.\.+", ",", text)          # Replace ... with comma pause
    text = re.sub(r"[^\w\s.,!?']", "", text)       # Remove unwanted characters
    text = re.sub(r"\s{2,}", " ", text)           # Remove double spaces
    if text and text[-1] not in ".!?":
        text += "."                               # Ensure it ends with punctuation
    return text


def generate_manga_voiceovers(
    manga_folder: str,
    output_subdir: str = "audio_output"
):
    manga_path = Path(manga_folder)
    bubbles_path = manga_path / "cleaned_bubbles.json"
    output_path = manga_path / output_subdir
    output_path.mkdir(exist_ok=True)

    if not bubbles_path.exists():
        raise FileNotFoundError(f"{bubbles_path} does not exist")

    with open(bubbles_path, "r", encoding="utf-8") as f:
        bubbles_by_image = json.load(f)

    engine = pyttsx3.init()

    # Optional: Adjust voice parameters here
    engine.setProperty('rate', 160)     # Speed (lower = slower, more natural)
    engine.setProperty('volume', 0.9)   # Volume (0.0 to 1.0)

    # List voices and choose one (optional)
    voices = engine.getProperty('voices')
    # engine.setProperty('voice', voices[1].id)  # Example: set to 2nd available voice

    for image_name, bubble in bubbles_by_image.items():
        mp3_name = f"{Path(image_name).stem}.mp3"
        audio_output_file = output_path / mp3_name

        text = bubble.get("text", "")
        cleaned = clean_text(text)
        if not cleaned.strip():
            continue

        print(f"[{image_name}] Generating speech...")

        engine.save_to_file(cleaned.strip(), str(audio_output_file))

    engine.runAndWait()
    print(f"\n✅ Voiceover generation complete. Output saved to: {output_path}")
