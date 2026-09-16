# FastVideo AI Studio Pro - Configuration Constants
# =================================================

import os
from typing import Tuple

# Supported Models Mapping: Friendly Name -> HuggingFace Repo ID or Local Path
MODEL_PATH_MAPPING = {
    "FastWan2.1-T2V-1.3B (480p - Siêu tốc)": "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
    "FastWan2.2-TI2V-5B (720p - Sparse VSA)": "FastVideo/FastWan2.2-TI2V-5B-Diffusers",
    "FastWan2.2-TI2V-5B-FullAttn (720p - Flash/SDPA)": "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
    # Legacy alias support:
    "FastWan2.1-T2V-1.3B": "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
    "FastWan2.2-TI2V-5B": "FastVideo/FastWan2.2-TI2V-5B-Diffusers",
}

DEFAULT_MODEL_NAME = "FastWan2.1-T2V-1.3B (480p - Siêu tốc)"

# Canonical Resolution Dropdown Choices
RESOLUTION_CHOICES = [
    "SD (480p - Gốc Siêu Tốc)",
    "HD (720p - Sắc Nét)",
    "Full HD (1080p - Siêu Nét)",
]
RES_SD = RESOLUTION_CHOICES[0]
RES_HD = RESOLUTION_CHOICES[1]
RES_FHD = RESOLUTION_CHOICES[2]

# Canonical Aspect Ratio Choices
ASPECT_RATIO_CHOICES = [
    "16:9 (Ngang - YouTube)",
    "9:16 (Dọc - Shorts/TikTok)",
    "1:1 (Vuông - Feed)",
]

# ---------------------------------------------------------------------------
# Resolution Configurations: (aspect_ratio, resolution) -> (native_w, native_h, target_w, target_h)
# ---------------------------------------------------------------------------

# 1. For 480P Base Models (e.g. FastWan2.1 1.3B):
RESOLUTION_CONFIGS_480P = {
    ("16:9 (Ngang)", RES_SD): (832, 448, 832, 448),
    ("16:9 (Ngang)", RES_HD): (832, 448, 1280, 720),
    ("16:9 (Ngang)", RES_FHD): (832, 448, 1920, 1080),
    ("9:16 (Dọc Shorts/Reels)", RES_SD): (448, 832, 448, 832),
    ("9:16 (Dọc Shorts/Reels)", RES_HD): (448, 832, 720, 1280),
    ("9:16 (Dọc Shorts/Reels)", RES_FHD): (448, 832, 1080, 1920),
    ("1:1 (Vuông)", RES_SD): (624, 624, 624, 624),
    ("1:1 (Vuông)", RES_HD): (624, 624, 720, 720),
    ("1:1 (Vuông)", RES_FHD): (624, 624, 1080, 1080),
}

# 2. For 720P Base Models (e.g. FastWan2.2 5B):
# Native 720p spatial dimensions must be divisible by 16 / 32 for VAE latent packing.
RESOLUTION_CONFIGS_720P = {
    ("16:9 (Ngang)", RES_SD): (832, 448, 832, 448),
    ("16:9 (Ngang)", RES_HD): (1280, 704, 1280, 720),
    ("16:9 (Ngang)", RES_FHD): (1280, 704, 1920, 1080),
    ("9:16 (Dọc Shorts/Reels)", RES_SD): (448, 832, 448, 832),
    ("9:16 (Dọc Shorts/Reels)", RES_HD): (704, 1280, 720, 1280),
    ("9:16 (Dọc Shorts/Reels)", RES_FHD): (704, 1280, 1080, 1920),
    ("1:1 (Vuông)", RES_SD): (624, 624, 624, 624),
    ("1:1 (Vuông)", RES_HD): (960, 960, 960, 960),
    ("1:1 (Vuông)", RES_FHD): (960, 960, 1080, 1080),
}

# Default backwards-compatible alias
RESOLUTION_CONFIGS = RESOLUTION_CONFIGS_480P


def get_resolution_config(model_name: str, aspect_ratio: str, resolution: str) -> Tuple[int, int, int, int]:
    """
    Dynamically select optimal native render and upscaling target based on model architecture.
    Robustly matches ratio (16:9, 9:16, 1:1) and resolution (480p/SD, 720p/HD, 1080p/FHD).
    """
    # 1. Normalize aspect ratio key
    if "9:16" in aspect_ratio:
        ratio_key = "9:16 (Dọc Shorts/Reels)"
    elif "1:1" in aspect_ratio:
        ratio_key = "1:1 (Vuông)"
    else:
        ratio_key = "16:9 (Ngang)"

    # 2. Normalize resolution key
    if "1080" in resolution or "full" in resolution.lower():
        res_key = RES_FHD
    elif "720" in resolution or "hd" in resolution.lower():
        res_key = RES_HD
    else:
        res_key = RES_SD

    is_720p_model = ("5B" in model_name or "720p" in model_name.lower() or "5b" in model_name.lower())
    configs = RESOLUTION_CONFIGS_720P if is_720p_model else RESOLUTION_CONFIGS_480P
    return configs.get((ratio_key, res_key), (832, 448, 832, 448))


# ---------------------------------------------------------------------------
# Duration & Frame Alignments (4k + 1 rule)
# ---------------------------------------------------------------------------
DURATION_CHOICES = ["3s", "4s", "5s", "6s (Đề xuất tối ưu)", "8s", "10s", "12s", "18s", "20s", "25s", "30s", "Tùy chỉnh"]

DURATION_TO_FRAMES = {
    "3s": 49,
    "4s": 61,
    "5s": 81,
    "6s (Đề xuất tối ưu)": 97,
    "6s": 97,
    "8s": 129,
    "10s": 161,
    "12s": 193,
    "18s": 289,
    "20s": 321,
    "25s": 401,
    "30s": 481,
}

MULTI_SCENE_DURATIONS = [
    "15s (3 phân cảnh x 5s)",
    "20s (4 phân cảnh x 5s)",
    "30s (6 phân cảnh x 5s)",
    "60s (12 phân cảnh x 5s)",
]

MULTI_SCENE_MAP = {
    "15s (3 phân cảnh x 5s)": 3,
    "20s (4 phân cảnh x 5s)": 4,
    "30s (6 phân cảnh x 5s)": 6,
    "60s (12 phân cảnh x 5s)": 12,
}

MAX_QUEUE_CARDS = 5
NUM_CARDS = 5
DEFAULT_FPS = 16
OUTPUTS_DIR = "outputs"
