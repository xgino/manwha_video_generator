
# Get manga images
def get_manwha_images():
    from manga_downloader import MangaDownloader
    url = "https://beginningaftertheendmanga.org/manga/the-beginning-after-the-end-chapter-1/"
    #url = "https://greatestestatedeveloper.org/manga/the-greatest-estate-developer-chapter-1/"

    downloader = MangaDownloader()
    downloader.download_chapter(url)





FOLDER = "downloads/the-beginning-after-the-end-chapter-2"

# Get Text Bubble and data
def get_textbubble_data():
    from get_image_data import run_on_folder
    run_on_folder(FOLDER)

# Get Text Bubble and data
def get_text_LLM_data():
    from llm_get_data import run_on_folder
    run_on_folder(FOLDER)

# Cleaning dict that is noicy
def clean_dict():
    from clean_dict import clean_bubble_data

    cleaned = clean_bubble_data(FOLDER)

    # Save to a new cleaned JSON file
    import json
    with open(f"{FOLDER}/cleaned_bubbles.json", "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print("🎉 Cleaned data saved to cleaned_bubbles.json")


# Get Voices
def get_voice():
    from voice import generate_manga_voiceovers
    generate_manga_voiceovers(FOLDER)




# GENERATE VIDEO
def generate_video():
    from create_video import generate_video
    generate_video(FOLDER, platform="tiktok")





get_text_LLM_data()