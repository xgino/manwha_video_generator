
# Get manga images
def get_manwha_images():
    from manga_downloader import MangaDownloader
    url = f"https://beginningaftertheendmanga.org/manga/the-beginning-after-the-end-chapter-{NUM}/"
    # url = "https://greatestestatedeveloper.org/manga/the-greatest-estate-developer-chapter-1/"

    downloader = MangaDownloader()
    downloader.download_chapter(url)

# Get Text Bubble and data
def get_textbubble_data(FOLDER):
    from get_image_data import run_on_folder
    run_on_folder(FOLDER)

# Get Text Bubble and data
def get_text_LLM_data(FOLDER):
    from llm_get_data import run_on_folder
    run_on_folder(FOLDER)

# Cleaning dict that is noicy
def clean_dict(FOLDER):
    from clean_dict import clean_bubble_data

    cleaned = clean_bubble_data(FOLDER)

    # Save to a new cleaned JSON file
    import json
    with open(f"{FOLDER}/cleaned_bubbles.json", "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print("🎉 Cleaned data saved to cleaned_bubbles.json")

# Get Voices
def get_voice(FOLDER):
    from voice import run_voiceover_generation
    run_voiceover_generation(FOLDER)

# GENERATE VIDEO
def generate_a_video(FOLDER, NUM):
    from generate_video import generate_video
    generate_video(FOLDER, NUM, platform="tiktok")




## Start script
import time

start = time.time()


def process_chapter(NUM):
    # Download Manga Images
    from manga_downloader import MangaDownloader

    url = f"https://beginningaftertheendmanga.org/manga/the-beginning-after-the-end-chapter-{NUM}/"
    FOLDER = f"downloads/the-beginning-after-the-end-chapter-{NUM}"
    # FOLDER = f"downloads/the-beginning-after-the-end-chapter-32"

    downloader = MangaDownloader()
    downloader.download_chapter(url)

    # Get Image Data
    get_text_LLM_data(FOLDER)

    # Get Voices
    get_voice(FOLDER)

    # GENERATE VIDEO
    generate_a_video(FOLDER, NUM)


# Loop through chapters 33 to 120
for NUM in range(56, 80):
    print(f"\n🔁 Processing chapter {NUM}")
    process_chapter(NUM)


end = time.time()
print(f"\n⏱️ Total script duration: {end - start:.2f} seconds")
