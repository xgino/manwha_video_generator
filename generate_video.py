import os
import cv2
import numpy as np
import random
import ffmpeg
from PIL import Image, ImageDraw, ImageFont


# === CONFIG ===
FRAME_RATE = 30
SPEECH_END_PADDING = 1.5
VIDEO_OUTPUT_DIR = "video_output"
PLATFORM_DIMENSIONS = {
    "youtube": (1920, 1080),
    "tiktok": (1080, 1920),
}
# CROSSFADE_FRAMES = int(0.5 * FRAME_RATE)
CROSSFADE_FRAMES = FRAME_RATE

def delete_long_audios(audio_folder, max_duration=15.0):
    deleted = 0
    for file in os.listdir(audio_folder):
        if file.lower().endswith(".mp3"):
            path = os.path.join(audio_folder, file)
            duration = get_audio_duration(path)
            if duration > max_duration:
                os.remove(path)
                print(f"[AUDIO] Deleted '{file}' (duration: {duration:.2f}s)")
                deleted += 1
    print(f"[AUDIO] {deleted} long audio file(s) deleted.")

# === UTILS ===
def get_sorted_files(folder, exts):
    return sorted([f for f in os.listdir(folder) if f.lower().split(".")[-1] in exts],
                  key=lambda x: int(os.path.splitext(x)[0]))


def get_audio_duration(audio_path):
    try:
        probe = ffmpeg.probe(audio_path)
        return float(probe['format']['duration'])
    except:
        return 2.0

def apply_fades(frames):
    faded = []
    total = len(frames)
    for i in range(CROSSFADE_FRAMES):
        alpha = i / CROSSFADE_FRAMES
        faded.append((frames[i].astype(np.float32) * alpha).astype(np.uint8))
    faded.extend(frames[CROSSFADE_FRAMES:-CROSSFADE_FRAMES])
    for i in range(CROSSFADE_FRAMES):
        alpha = 1 - (i / CROSSFADE_FRAMES)
        faded.append((frames[-CROSSFADE_FRAMES + i].astype(np.float32) * alpha).astype(np.uint8))
    return faded

def pan_crop_effect(img, target_w, target_h, frames, direction):
    h, w = img.shape[:2]
    if w < target_w or h < target_h:
        scale = max(target_w / w, target_h / h)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
        h, w = img.shape[:2]

    zoom_start = 1.0
    zoom_end = random.uniform(0.9, 1.1)

    clips = []
    for i in range(frames):
        t = i / (frames - 1)
        smooth_t = t * t * (3 - 2 * t)
        scale = zoom_start + (zoom_end - zoom_start) * smooth_t

        crop_w = min(int(target_w / scale), w)
        crop_h = min(int(target_h / scale), h)

        if direction == "left_to_right":
            x = int((w - crop_w) * smooth_t)
            y = (h - crop_h) // 2
        elif direction == "top_to_bottom":
            y = int((h - crop_h) * smooth_t)
            x = (w - crop_w) // 2
        else:
            x = (w - crop_w) // 2
            y = (h - crop_h) // 2

        cropped = img[y:y+crop_h, x:x+crop_w]
        frame = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        clips.append(frame)
    return clips

def save_video_chunk(frames, output_path, frame_size):
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), FRAME_RATE, frame_size)
    for frame in frames:
        out.write(cv2.resize(frame, frame_size))
    out.release()

# === STEP 1: Generate video from images (no audio) ===
def generate_video_chunks(image_dir, platform="youtube"):
    video_dir = os.path.join(image_dir, "video_output")
    os.makedirs(video_dir, exist_ok=True)

    width, height = PLATFORM_DIMENSIONS[platform.lower()]
    frame_size = (width, height)
    audio_dir = os.path.join(image_dir, "audio_output")

    images = get_sorted_files(image_dir, {"jpg", "jpeg", "png"})

    for idx, img_name in enumerate(images):
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(image_dir, img_name)
        audio_path = os.path.join(audio_dir, f"{base_name}.mp3")

        has_audio = os.path.exists(audio_path)
        duration = get_audio_duration(audio_path) + SPEECH_END_PADDING if has_audio else 5.0
        frame_count = int(duration * FRAME_RATE)

        img = np.array(Image.open(img_path).convert("RGB"))[:, :, ::-1]
        direction = "left_to_right" if width > height else "top_to_bottom"

        frames = pan_crop_effect(img, width, height, frame_count, direction)
        # frames = apply_fades(frames)

        output_path = os.path.join(video_dir, f"{base_name}.mp4")
        save_video_chunk(frames, output_path, frame_size)
        print(f"[VIDEO] {output_path} generated ({duration:.2f}s)")


def ensure_audio_files_exist(folder_path):
    video_dir = os.path.join(folder_path, "video_output")
    audio_dir = os.path.join(folder_path, "audio_output")
    os.makedirs(audio_dir, exist_ok=True)

    video_files = [
        f for f in os.listdir(video_dir)
        if f.lower().endswith(".mp4")
    ]

    for video_name in video_files:
        base = os.path.splitext(video_name)[0]
        audio_path = os.path.join(audio_dir, f"{base}.mp3")

        if not os.path.exists(audio_path):
            silent_path = audio_path
            (
                ffmpeg
                .input("anullsrc=channel_layout=stereo:sample_rate=44100", f='lavfi', t=5)
                .output(silent_path, acodec='mp3')
                .overwrite_output()
                .run(quiet=True)
            )
            print(f"[INFO] Created silent audio: {silent_path}")

# === STEP 2: Add audio to video ===
def combine_videos_with_audio(image_dir):
    video_dir = os.path.join(image_dir, "video_output")
    audio_dir = os.path.join(image_dir, "audio_output")
    temp_dir = os.path.join(image_dir, "temp_final_chunks")
    os.makedirs(temp_dir, exist_ok=True)

    video_files = get_sorted_files(video_dir, {"mp4"})
    merged_files = []

    if not video_files:
        print("[ERROR] No video files found.")
        return

    for video_name in video_files:
        base = os.path.splitext(video_name)[0]
        video_path = os.path.join(video_dir, video_name)
        audio_path = os.path.join(audio_dir, f"{base}.mp3")

        if not os.path.exists(audio_path):
            print(f"[SKIP] No matching audio for {video_name}")
            continue

        reencoded_audio_path = os.path.join(temp_dir, f"{base}_audio.aac")
        output_path = os.path.join(temp_dir, f"{base}_final.mp4")

        # Re-encode audio to AAC
        # (
        #     ffmpeg
        #     .input(audio_path)
        #     .output(reencoded_audio_path, acodec="aac", ar="44100", ac=2)
        #     .overwrite_output()
        #     .run(quiet=True)
        # )

        # Combine video + re-encoded audio
        (
            ffmpeg
            .output(
                ffmpeg.input(video_path),
                #ffmpeg.input(reencoded_audio_path),
                output_path,
                vcodec="libx264",
                pix_fmt="yuv420p",
                acodec="aac",
                shortest=None
            )
            .overwrite_output()
            .run(quiet=True)
        )

        merged_files.append(output_path)
        print(f"[MERGED] {output_path}")

    if not merged_files:
        print("[ERROR] No merged files to concatenate.")
        return

    # Create concat list file
    concat_list_path = os.path.join(temp_dir, "concat_list.txt")
    with open(concat_list_path, "w") as f:
        for fpath in merged_files:
            f.write(f"file '{os.path.abspath(fpath)}'\n")

    final_output = os.path.join(image_dir, "combined.mp4")

    # Concatenate final video
    (
        ffmpeg
        .input(concat_list_path, format='concat', safe=0)
        .output(final_output, c='copy')
        .overwrite_output()
        .run(quiet=True)
    )

    print(f"[FINAL VIDEO] {final_output} created.")


def get_video_resolution(video_path):
    probe = ffmpeg.probe(video_path)
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    return int(video_stream["width"]), int(video_stream["height"])

def add_intro_and_credits(folder_path, cover_path, NUM, CREDITS):
    # Your final video path
    main_video = os.path.join(folder_path, "combined.mp4")
    
    temp_dir = os.path.join(folder_path, "intro_temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Get resolution of final video
    width, height = get_video_resolution(main_video)

    ## --------------------------------------------
    ## STEP 1 - Create Cover Image resized properly
    ## --------------------------------------------

    cover_img = Image.open(cover_path).convert("RGB")
    
    # Resize cover image to cover entire video frame keeping aspect ratio
    cover_ratio = cover_img.width / cover_img.height
    target_ratio = width / height
    
    if cover_ratio > target_ratio:
        # Image is wider → resize height
        new_height = height
        new_width = int(height * cover_ratio)
    else:
        # Image is taller → resize width
        new_width = width
        new_height = int(width / cover_ratio)
    
    cover_img_resized = cover_img.resize((new_width, new_height), Image.LANCZOS)
    
    # Crop center
    left = (new_width - width) // 2
    top = (new_height - height) // 2
    cover_img_cropped = cover_img_resized.crop(
        (left, top, left + width, top + height)
    )
    
    # Add NUM text bottom-right with small padding
    draw = ImageDraw.Draw(cover_img_cropped)

    text = str(NUM)
    padding = int(height * 0.03)
    font_size = int(height * 0.08)

    try:
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", font_size)
    except Exception as e:
        print(f"[WARNING] Falling back to default font: {e}")
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Ensure text is fully inside the image
    x = max(0, width - text_w - padding)
    y = max(0, height - text_h - padding)

    # Draw white text on top
    draw.text((x, y), text, fill="white", font=font)
    
    cover_img_path = os.path.join(temp_dir, "cover_temp.jpg")
    cover_img_cropped.save(cover_img_path)

    # Convert cover image to mp4 (1 second, silent)
    cover_mp4_path = os.path.join(temp_dir, "cover.mp4")
    ffmpeg.input(cover_img_path, loop=1, t=1) \
        .filter("scale", width, height) \
        .output(cover_mp4_path, pix_fmt="yuv420p", vcodec="libx264", r=30, shortest=None) \
        .overwrite_output() \
        .run()
    
    ## ----------------------------------------
    ## STEP 2 - Create Credits Image
    ## ----------------------------------------

    # Prepare credits text block
    credit_text = "\n".join([f"{k}: {v}" for k, v in CREDITS.items()])
    
    # Create blank black image
    credits_img = Image.new("RGB", (width, height), color="black")
    draw = ImageDraw.Draw(credits_img)
    
    # Determine max text width and height
    max_width = width * 0.9
    
    font_size = int(height * 0.02)
    
    try:
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", font_size)
    except Exception as e:
        print(f"[WARNING] Falling back to default font: {e}")
        font = ImageFont.load_default()

    # Split credits into lines
    lines = credit_text.split("\n")
    
    # Measure height of all lines
    line_sizes = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        line_sizes.append((w, h))

    total_text_height = sum(h for (_, h) in line_sizes) + (len(lines)-1)*10
    
    y_start = (height - total_text_height) // 2
    
    # Draw each line centered
    y = y_start
    for line, (w, h) in zip(lines, line_sizes):
        x = (width - w) // 2
        draw.text((x, y), line, fill="white", font=font)
        y += h + 10
    
    credits_img_path = os.path.join(temp_dir, "credits_temp.jpg")
    credits_img.save(credits_img_path)

    # Convert credits image to mp4 (2 seconds, silent)
    credits_mp4_path = os.path.join(temp_dir, "credits.mp4")
    ffmpeg.input(credits_img_path, loop=1, t=2) \
        .filter("scale", width, height) \
        .output(credits_mp4_path, pix_fmt="yuv420p", vcodec="libx264", r=30, shortest=None) \
        .overwrite_output() \
        .run()
    
    ## ----------------------------------------
    ## STEP 3 - Concatenate All Videos
    ## ----------------------------------------
    
    # All videos must have same codec, pixel format, resolution, fps
    # Concat in order: cover → credits → main
    concat_list = os.path.join(temp_dir, "concat.txt")
    with open(concat_list, "w") as f:
        f.write(f"file '{os.path.abspath(cover_mp4_path)}'\n")
        f.write(f"file '{os.path.abspath(credits_mp4_path)}'\n")
        f.write(f"file '{os.path.abspath(main_video)}'\n")
    
    output_path = os.path.join(folder_path, "final_intro.mp4")
    
    (
        ffmpeg
        .input(concat_list, format="concat", safe=0)
        .output(output_path, c="copy", acodec="aac")
        .overwrite_output()
        .run(quiet=True)
    )
    
    return output_path

def add_background_music(folder_path, music_path, music_volume=0.1):
    video_path = os.path.join(folder_path, "final_intro.mp4")
    final_output = os.path.join(folder_path, "final.mp4")

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    try:
        # Get video duration
        video_info = ffmpeg.probe(video_path)
        video_duration = float(next(stream for stream in video_info['streams'] if stream['codec_type'] == 'video')['duration'])

        # Loop background music until it matches video length
        music_input = ffmpeg.input(music_path, stream_loop=-1)
        music_audio = music_input.audio.filter('volume', music_volume)

        video_input = ffmpeg.input(video_path)

        # Check if original video has audio
        has_audio = any(stream['codec_type'] == 'audio' for stream in video_info['streams'])

        if has_audio:
            # Mix original video audio and background music
            mixed_audio = ffmpeg.filter([video_input.audio, music_audio], 'amix', inputs=2, duration='first', dropout_transition=3)
        else:
            # Use only background music
            mixed_audio = music_audio

        out = (
            ffmpeg
            .output(video_input.video, mixed_audio, final_output,
                    vcodec='copy', acodec='aac', audio_bitrate='192k', shortest=None)
            .overwrite_output()
        )

        out.run()
        print(f"Final video saved to: {final_output}")

    except ffmpeg.Error as e:
        print("FFmpeg error:", e.stderr.decode())


import shutil
def clean_temp(folder_path):
    targets = ['temp_final_chunks', 'intro_temp']

    for name in targets:
        dir_path = os.path.join(folder_path, name)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"Deleted: {dir_path}")
            except Exception as e:
                print(f"Error deleting {dir_path}: {e}")
        else:
            print(f"Not found or not a directory: {dir_path}")

# === MAIN ===
def generate_video(folder_path, NUM, platform="youtube"):
    audio_dir = os.path.join(folder_path, "audio_output")
    delete_long_audios(audio_dir, max_duration=15.0)
    generate_video_chunks(folder_path, platform)
    ensure_audio_files_exist(folder_path)
    combine_videos_with_audio(folder_path)
    
    AUDIO_CREDITS = {
        "./background_audio/rainy-lofi-city-lofi-music-332746.mp3": "Music by kaveesha Senanayake from Pixabay",
        "./background_audio/coffee-lofi-chill-lofi-music-332738.mp3": "Music by kaveesha Senanayake from Pixabay",
        "./background_audio/lofi-girl-lofi-background-music-361058.mp3": "Music by kaveesha Senanayake from Pixabay",
    }

    def select_random_audio(audio_dir="./background_audio"):
        files = [f for f in os.listdir(audio_dir) if f.endswith(".mp3")]
        if not files:
            raise FileNotFoundError("No .mp3 files found in background_audio/")
        selected = random.choice(files)
        full_path = os.path.join(audio_dir, selected)
        credit = AUDIO_CREDITS.get(full_path, "Audio by Unknown")
        return full_path, credit
    
    music_path, music_credit = select_random_audio()

    CREDITS = {
        "Written by": "Yūgo Kobayashi",
        "Written by": "Naohiko Ueno",
        "Illustrated by": "Yūgo Kobayashi",
        "Published by": "Shogakukan",
        "publisher": "SG: Shogakukan Asia",
        "Music": music_credit
    }

    cover_path = "./cover/aoshi.jpg"
    add_intro = add_intro_and_credits(folder_path, cover_path, NUM, CREDITS)

    add_background_music(folder_path, music_path)

    clean_temp(folder_path)
