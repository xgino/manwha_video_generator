import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin


class MangaDownloader:
    def __init__(self, base_dir="downloads"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def download_chapter(self, url: str) -> str:
        print(f"[+] Downloading chapter from: {url}")
        chapter_name = self._get_chapter_name(url)
        chapter_folder = os.path.join(self.base_dir, chapter_name)
        os.makedirs(chapter_folder, exist_ok=True)

        soup = self._get_soup(url)
        image_urls = self._extract_image_urls(soup, base_url=url)

        if not image_urls:
            print("[-] No images found. Trying fallback methods...")

        for idx, img_url in enumerate(image_urls, start=1):
            filename = f"{idx}.jpg"
            filepath = os.path.join(chapter_folder, filename)
            try:
                self._download_image(img_url, filepath)
                print(f"  -> Saved page {idx} to {filepath}")
            except Exception as e:
                print(f"[!] Failed to download {img_url}: {e}")

        print(f"[✓] Chapter saved to: {chapter_folder}")
        return chapter_folder

    def _get_chapter_name(self, url: str) -> str:
        path = urlparse(url).path
        parts = [p for p in path.split("/") if p]
        return parts[-1] if parts else "chapter"

    def _get_soup(self, url: str) -> BeautifulSoup:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return BeautifulSoup(response.content, "html.parser")

    def _extract_image_urls(self, soup: BeautifulSoup, base_url: str) -> list:
        image_urls = []

        # Try container with multiple images
        container = soup.select_one("div.container-chapter-reader")
        if container:
            img_tags = container.find_all("img")
        else:
            img_tags = soup.find_all("img")  # fallback

        for img in img_tags:
            for attr in ["data-src", "data-lazy-src", "src"]:
                img_url = img.get(attr)
                if img_url and img_url.startswith(("http", "//")):
                    # Normalize URL (add http: if needed)
                    if img_url.startswith("//"):
                        img_url = "https:" + img_url
                    image_urls.append(img_url)
                    break  # only take first valid one

        return image_urls

    def _download_image(self, url: str, save_path: str):
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
