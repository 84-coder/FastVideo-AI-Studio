# FastVideo AI Studio Pro - Video Postprocessing & Enhancement Engine

import os
import cv2
import imageio


def upscale_video_file(input_path: str, output_path: str, target_width: int, target_height: int, fps: int = 16) -> str:
    """
    Upscale video using Lanczos4 interpolation + Unsharp Masking sharpening.
    Yields crisp 720p / 1080p output without neural artifacting.
    """
    try:
        reader = imageio.get_reader(input_path)
        writer = imageio.get_writer(output_path, fps=fps, format="mp4", codec="libx264", macro_block_size=1)
        for frame in reader:
            resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            gaussian = cv2.GaussianBlur(resized, (0, 0), 1.5)
            enhanced = cv2.addWeighted(resized, 1.25, gaussian, -0.25, 0)
            writer.append_data(enhanced)
        reader.close()
        writer.close()
        return output_path
    except Exception as e:
        print(f"Upscaling warning: {e}, falling back to original video file.")
        return input_path


def concatenate_video_files(video_paths: list[str], output_path: str, fps: int = 16) -> str:
    """
    Losslessly stitch together multiple sequential MP4 video clips into one cohesive continuous film.
    """
    valid_paths = [p for p in video_paths if p and os.path.exists(p)]
    if not valid_paths:
        raise FileNotFoundError("No valid video files to concatenate.")

    writer = imageio.get_writer(output_path, fps=fps, format="mp4", codec="libx264", macro_block_size=1)
    for v_path in valid_paths:
        reader = imageio.get_reader(v_path)
        for frame in reader:
            writer.append_data(frame)
        reader.close()
    writer.close()
    return output_path


def extract_last_frame(video_path: str, output_image_path: str) -> str | None:
    """
    Extract the very last frame of a video clip for seed conditioning.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            cv2.imwrite(output_image_path, frame)
            return output_image_path
    except Exception as e:
        print(f"Extract last frame error: {e}")
    return None


def parse_multi_scenes(prompt_text: str, num_scenes: int) -> list[str]:
    """
    Parse a studio multi-scene prompt:
    - If user separated with '---', use their custom scenes.
    - Otherwise, automatically generate progressive cinematic camera directions for consecutive 5s shots.
    """
    if "---" in prompt_text:
        parts = [p.strip() for p in prompt_text.split("---") if p.strip()]
        if len(parts) >= num_scenes:
            return parts[:num_scenes]
    else:
        parts = [prompt_text.strip()]

    camera_moves = [
        "wide cinematic establishing shot, sweeping landscape, golden hour lighting",
        "slow smooth medium pan, cinematic depth of field, detailed textures",
        "dramatic low-angle push in, epic motion, cinematic atmosphere",
        "close-up subject tracking, smooth parallax movement, soft lighting",
        "elevated aerial drone tracking shot, dynamic perspective, cinematic grade",
        "climax slow-motion cinematic focus, high contrast, film grain",
    ]

    base = parts[0]
    scenes = list(parts)
    while len(scenes) < num_scenes:
        idx = len(scenes)
        move = camera_moves[(idx - 1) % len(camera_moves)]
        scenes.append(f"{base}, scene {idx+1}, {move}")
    return scenes
