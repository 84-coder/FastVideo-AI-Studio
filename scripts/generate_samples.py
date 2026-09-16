"""
Generate Showcase Videos for FastVideo AI Studio Pro README.
============================================================
Creates:
1. 16:9 Landscape: Golden eagle soaring over snowy mountain peaks (Cinematic)
2. 9:16 Portrait: Cute red panda drinking hot chocolate (Shorts/TikTok)
Both .mp4 and animated .gif versions are saved to assets/videos/.
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("FASTVIDEO_LOOPBACK_IP", "127.0.0.1")
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")
os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")

import time
import shutil
from copy import deepcopy
import torch
import cv2
import imageio

from fastvideo.entrypoints.video_generator import VideoGenerator
from fastvideo.api.sampling_param import SamplingParam


def convert_mp4_to_gif(mp4_path: str, gif_path: str, target_width: int = 400, fps: int = 12):
    """Convert an MP4 video to an optimized animated GIF for GitHub README."""
    print(f"[CONVERT] Converting {mp4_path} -> {gif_path} (width={target_width}, fps={fps})...")
    reader = imageio.get_reader(mp4_path)
    meta = reader.get_meta_data()
    orig_fps = meta.get("fps", 16)
    step = max(1, int(round(orig_fps / fps)))

    frames = []
    for i, frame in enumerate(reader):
        if i % step == 0:
            h, w, _ = frame.shape
            new_h = int(h * (target_width / w))
            # Ensure even dimensions
            new_h = (new_h // 2) * 2
            resized = cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)
            frames.append(resized)
    reader.close()

    duration = 1.0 / fps
    imageio.mimsave(gif_path, frames, format="GIF", duration=duration, loop=0)
    print(f"[OK] Created GIF: {gif_path} ({os.path.getsize(gif_path) / 1024 / 1024:.2f} MB)")


def main():
    model_path = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
    assets_dir = os.path.abspath("assets/videos")
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print(f"[LOAD] Loading generator with model: {model_path}...")
    print("=" * 60)

    generator = VideoGenerator.from_pretrained(
        model_path,
        num_gpus=1,
        text_encoder_cpu_offload=True,
        dit_layerwise_offload=False,
        dit_cpu_offload=True,
        vae_cpu_offload=True,
    )
    base_param = SamplingParam.from_pretrained(model_path)

    samples = [
        {
            "name": "sample_landscape_eagle",
            "prompt": "A majestic golden eagle soaring over snowy mountain peaks during a vibrant crimson sunset, 8k resolution, cinematic lighting, photorealistic.",
            "width": 832,
            "height": 448,
            "num_frames": 61,
            "seed": 42,
            "gif_width": 480,
        },
        {
            "name": "sample_portrait_panda",
            "prompt": "A cute red panda wearing a tiny knitted sweater drinking hot chocolate by a cozy wooden fireplace in a winter cabin, cinematic warm atmosphere.",
            "width": 448,
            "height": 832,
            "num_frames": 61,
            "seed": 1024,
            "gif_width": 320,
        },
    ]

    for idx, item in enumerate(samples):
        print("\n" + "=" * 60)
        print(f"[RUN] [{idx+1}/{len(samples)}] Generating: {item['name']}")
        print(f"[PROMPT] {item['prompt']}")
        print(f"[CONFIG] {item['width']}x{item['height']} | Frames: {item['num_frames']}")
        print("=" * 60)

        param = deepcopy(base_param)
        param.prompt = item["prompt"]
        param.width = item["width"]
        param.height = item["height"]
        param.num_frames = item["num_frames"]
        param.seed = item["seed"]
        param.guidance_scale = 3.0

        t0 = time.time()
        res = generator.generate_video(
            prompt=item["prompt"],
            sampling_param=param,
            save_video=True,
            return_frames=False,
        )
        elapsed = time.time() - t0
        print(f"[DONE] Video generated in {elapsed:.1f} seconds!")

        src_video = res.get("video_path")
        if not src_video or not os.path.exists(src_video):
            # Fallback to newest mp4 in outputs
            mp4s = [os.path.join("outputs", f) for f in os.listdir("outputs") if f.endswith(".mp4")]
            if mp4s:
                src_video = max(mp4s, key=os.path.getmtime)

        dst_mp4 = os.path.join(assets_dir, f"{item['name']}.mp4")
        dst_gif = os.path.join(assets_dir, f"{item['name']}.gif")

        if src_video and os.path.exists(src_video):
            shutil.copy2(src_video, dst_mp4)
            print(f"[SAVE] Saved MP4 to: {dst_mp4}")
            convert_mp4_to_gif(dst_mp4, dst_gif, target_width=item["gif_width"], fps=12)
        else:
            print(f"[ERROR] Output video file not found for {item['name']}")

        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("[ALL DONE] ALL SAMPLES GENERATED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
