import os
import cv2
import numpy as np
import random
import ffmpeg
from PIL import Image

# === CONFIG ===
FRAME_RATE = 30
SPEECH_END_PADDING = 1.2
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
                vcodec="copy",
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

    final_output = os.path.join(image_dir, "final.mp4")

    # Concatenate final video
    (
        ffmpeg
        .input(concat_list_path, format='concat', safe=0)
        .output(final_output, c='copy')
        .overwrite_output()
        .run(quiet=True)
    )

    print(f"[FINAL VIDEO] {final_output} created.")


# def add_lofi_background_audio(image_dir, lofi_audio_path):
#     video_path = os.path.join(image_dir, "final.mp4")
#     output_path = os.path.join(image_dir, "final_bg.mp4")

#     # Load inputs
#     video_input = ffmpeg.input(video_path)
#     lofi_input = ffmpeg.input(lofi_audio_path)

#     # Adjust lofi volume
#     lofi_audio = lofi_input.audio.filter("volume", 0.3)
#     video_audio = video_input.audio

#     # Mix both
#     mixed_audio = ffmpeg.filter(
#         [video_audio, lofi_audio],
#         "amix",
#         inputs=2,
#         duration="shortest",
#         dropout_transition=2
#     )

#     # Combine video stream + mixed audio
#     (
#         ffmpeg
#         .output(video_input["v"], mixed_audio, output_path, vcodec="copy", acodec="aac")
#         .overwrite_output()
#         .run(quiet=True)
#     )

#     print(f"[LOFI] Final with background lofi: {output_path}")

# === MAIN ===
def generate_video(folder_path, platform="youtube"):
    audio_dir = os.path.join(folder_path, "audio_output")
    delete_long_audios(audio_dir, max_duration=15.0)
    generate_video_chunks(folder_path, platform)
    ensure_audio_files_exist(folder_path)
    combine_videos_with_audio(folder_path)
    # lofi_path = "lofi.mp3"  # path to your lofi music
    # add_lofi_background_audio(folder_path, lofi_path)

