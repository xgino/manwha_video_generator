import os
import cv2
import random
import numpy as np
import ffmpeg
from PIL import Image
from time import sleep

# Platform dimensions
PLATFORM_DIMENSIONS = {
    "youtube": (1920, 1080),
    "tiktok": (1080, 1920),
    "instagram": (1080, 1080),
}

# Config
IMAGE_DURATION_NO_AUDIO = 2  # seconds (a bit faster than with audio)
SPEECH_END_PADDING = 1       # seconds to wait after audio
SPEECH_SPEED = 0.6           # 1.0 is normal, <1 slower, >1 faster
ZOOM_SPEED_FACTOR = 0.5      # Zoom transitions are slower
FRAME_RATE = 30              # FPS
TEMP_FRAME_DIR = "temp_frames"

def get_sorted_files(folder, exts):
    return sorted(
        [f for f in os.listdir(folder) if f.lower().split(".")[-1] in exts],
        key=lambda x: int(os.path.splitext(x)[0])
    )

def pan_crop_effect(img, target_w, target_h, frames, direction):
    h, w = img.shape[:2]
    clips = []

    # Upscale small images to cover the frame area
    if w < target_w or h < target_h:
        scale = max(target_w / w, target_h / h)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
        h, w = img.shape[:2]

    for i in range(frames):
        t = i / (frames - 1)
        # Ease-in-out
        smooth_t = t * t * (3 - 2 * t)

        if direction == "left_to_right" and w > target_w:
            max_x = w - target_w
            x = int(max_x * smooth_t)
            y = (h - target_h) // 2
        elif direction == "top_to_bottom" and h > target_h:
            max_y = h - target_h
            y = int(max_y * smooth_t)
            x = (w - target_w) // 2
        elif direction == "zoom_in":
            zoom_t = smooth_t * ZOOM_SPEED_FACTOR
            scale = 1 + 0.1 * zoom_t
            crop_w = int(target_w / scale)
            crop_h = int(target_h / scale)
            crop_w = min(crop_w, w)
            crop_h = min(crop_h, h)
            x = (w - crop_w) // 2
            y = (h - crop_h) // 2
            cropped = img[y:y+crop_h, x:x+crop_w]
            frame = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            clips.append(frame)
            continue
        else:
            # No movement if pan not supported
            x = (w - target_w) // 2
            y = (h - target_h) // 2

        cropped = img[y:y+target_h, x:x+target_w]
        clips.append(cropped)

    return clips

def pan_crop_effect(img, target_w, target_h, frames, direction):
    h, w = img.shape[:2]
    clips = []

    # Upscale small images to cover the frame area
    if w < target_w or h < target_h:
        scale = max(target_w / w, target_h / h)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
        h, w = img.shape[:2]

    for i in range(frames):
        t = i / (frames - 1)
        # Ease-in-out
        smooth_t = t * t * (3 - 2 * t)

        if direction == "left_to_right" and w > target_w:
            max_x = w - target_w
            x = int(max_x * smooth_t)
            y = (h - target_h) // 2
        elif direction == "top_to_bottom" and h > target_h:
            max_y = h - target_h
            y = int(max_y * smooth_t)
            x = (w - target_w) // 2
        elif direction == "zoom_in":
            zoom_t = smooth_t * ZOOM_SPEED_FACTOR
            scale = 1 + 0.1 * zoom_t
            crop_w = int(target_w / scale)
            crop_h = int(target_h / scale)
            crop_w = min(crop_w, w)
            crop_h = min(crop_h, h)
            x = (w - crop_w) // 2
            y = (h - crop_h) // 2
            cropped = img[y:y+crop_h, x:x+crop_w]
            frame = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            clips.append(frame)
            continue
        else:
            # No movement if pan not supported
            x = (w - target_w) // 2
            y = (h - target_h) // 2

        cropped = img[y:y+target_h, x:x+target_w]
        clips.append(cropped)

    return clips

def generate_video(folder_path, platform="youtube"):
    platform = platform.lower()
    if platform not in PLATFORM_DIMENSIONS:
        raise ValueError(f"Unsupported platform '{platform}'")

    width, height = PLATFORM_DIMENSIONS[platform]
    frame_size = (width, height)
    audio_folder = os.path.join(folder_path, "audio_output")
    os.makedirs(TEMP_FRAME_DIR, exist_ok=True)

    images = get_sorted_files(folder_path, {"jpg", "jpeg", "png"})
    audios = get_sorted_files(audio_folder, {"mp3", "wav", "aac"}) if os.path.exists(audio_folder) else []

    video_chunks = []

    for idx, img_file in enumerate(images):
        img_path = os.path.join(folder_path, img_file)
        audio_file = f"{os.path.splitext(img_file)[0]}.mp3"
        audio_path = os.path.join(audio_folder, audio_file)
        has_audio = os.path.exists(audio_path)

        if has_audio:
            duration = get_audio_duration(audio_path) / SPEECH_SPEED + SPEECH_END_PADDING
        else:
            duration = IMAGE_DURATION_NO_AUDIO

        frame_count = int(duration * FRAME_RATE)

        img = np.array(Image.open(img_path).convert("RGB"))[:, :, ::-1]  # to BGR

        if platform == "tiktok":
            direction = random.choice(["top_to_bottom", "zoom_in"])
        else:
            direction = random.choice(["left_to_right", "zoom_in"])

        frames = pan_crop_effect(img, width, height, frame_count, direction)

        chunk_basename = f"chunk_{idx}"
        chunk_path = os.path.join(TEMP_FRAME_DIR, chunk_basename + ".mp4")
        save_video_chunk(frames, chunk_path, frame_size)

        if has_audio:
            chunk_with_audio = os.path.join(TEMP_FRAME_DIR, chunk_basename + "_audio.mp4")
            merge_audio(chunk_path, audio_path, chunk_with_audio)
            video_chunks.append(chunk_with_audio)
        else:
            video_chunks.append(chunk_path)

    final_output = os.path.join(folder_path, f"{os.path.basename(folder_path)}_{platform}.mp4")
    concatenate_videos(video_chunks, final_output)
    print(f"✅ Video exported: {final_output}")

def get_audio_duration(audio_path):
    try:
        probe = ffmpeg.probe(audio_path)
        return float(probe['format']['duration'])
    except:
        return IMAGE_DURATION_NO_AUDIO

def save_video_chunk(frames, output_path, frame_size):
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), FRAME_RATE, frame_size)
    for frame in frames:
        out.write(frame)
    out.release()

def merge_audio(video_path, audio_path, output_path):
    video_input = ffmpeg.input(video_path)
    audio_input = ffmpeg.input(audio_path)
    
    (
        ffmpeg
        .output(video_input, audio_input, output_path, vcodec='copy', acodec='aac', strict='experimental')
        .overwrite_output()
        .run(quiet=True)
    )

def concatenate_videos(video_paths, output_path):
    txt_path = os.path.join(TEMP_FRAME_DIR, "concat_list.txt")
    with open(txt_path, "w") as f:
        for path in video_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")

    (
        ffmpeg
        .input(txt_path, format="concat", safe=0)
        .output(output_path, c="copy")
        .overwrite_output()
        .run(quiet=True)
    )
