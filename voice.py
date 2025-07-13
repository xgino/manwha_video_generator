import json
from pathlib import Path
import re
import asyncio
import edge_tts


def clean_text(text: str) -> str:
    """
    Clean text for smoother TTS reading.
    """
    text = text.replace("\n", " ").strip()
    text = re.sub(r"--+", ",", text)
    text = re.sub(r"\.\.\.+", ",", text)
    text = re.sub(r"[^\w\s.,!?']", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    if text and text[-1] not in ".!?":
        text += "."
    return text


def extract_numeric_part(image_path: str) -> int:
    """
    Extract numeric part from image filename.
    E.g. "downloads/.../10.jpg" → 10
    """
    if not image_path:
        return -1
    stem = Path(image_path).stem
    match = re.search(r"(\d+)", stem)
    return int(match.group(1)) if match else -1


async def generate_manga_voiceovers(
    manga_folder: str,
    output_subdir: str = "audio_output",
    voice: str = "en-US-AriaNeural",
    rate: str = "+0%",
    volume: str = "+0%"
):
    """
    Generate audio voiceovers for manga JSON data using edge-tts.
    
    Args:
        manga_folder: Path to folder containing JSON files or a single JSON file.
        output_subdir: Where to save MP3 files.
        voice: Name of the neural voice (Edge TTS voice list).
        rate: Speaking rate adjustment (e.g. "+10%").
        volume: Volume adjustment (e.g. "+3%").
    """
    manga_path = Path(manga_folder)
    output_path = manga_path / output_subdir
    output_path.mkdir(parents=True, exist_ok=True)

    # If folder contains multiple JSONs, loop them all
    if manga_path.is_dir():
        json_files = list(manga_path.glob("*.json"))
    else:
        json_files = [manga_path]

    for json_file in json_files:
        print(f"🔎 Processing {json_file} ...")
        with open(json_file, "r", encoding="utf-8") as f:
            # Either single dict or list of dicts
            data = json.load(f)

        # Wrap single object into list
        if isinstance(data, dict):
            data = [data]

        # Sort entries by numeric part of filename
        data_sorted = sorted(
            data,
            key=lambda entry: extract_numeric_part(entry.get("image", ""))
        )

        for idx, entry in enumerate(data_sorted, start=1):
            image_path = entry.get("image")
            cleaned_text = entry.get("cleaned_text")
            extracted_text = entry.get("extracted_text")

            text = cleaned_text or extracted_text or ""
            text = clean_text(text)
            if not text.strip():
                print(f"⚠️ Skipping empty text for entry {idx}")
                continue

            # Determine output filename
            image_filename = Path(image_path).stem if image_path else f"voice_{idx}"
            mp3_filename = f"{image_filename}.mp3"
            mp3_path = output_path / mp3_filename

            print(f"🎙️ Generating voiceover for [{image_filename}]")

            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate,
                volume=volume,
            )
            await communicate.save(str(mp3_path))

    print(f"\n✅ All voiceovers saved to: {output_path}")


def run_voiceover_generation(FOLDER):
    """
    Example runner function.
    """
    asyncio.run(
        generate_manga_voiceovers(
            manga_folder=FOLDER,
            output_subdir="audio_output",
            voice="en-US-AriaNeural",
            rate="+0%",
            volume="+0%",
        )
    )
