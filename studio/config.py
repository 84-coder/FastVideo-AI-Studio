# FastVideo AI Studio Pro - Configuration Constants

import os

MODEL_PATH_MAPPING = {
    "FastWan2.1-T2V-1.3B": "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
}

DEFAULT_MODEL_NAME = "FastWan2.1-T2V-1.3B"

# Aspect Ratio and Resolution Mapping:
# (aspect_ratio, resolution): (native_w, native_h, target_w, target_h)
RESOLUTION_CONFIGS = {
    ("16:9 (Ngang)", "SD (480p - Gốc siêu nhanh)"): (832, 448, 832, 448),
    ("16:9 (Ngang)", "HD (720p - Sắc nét)"): (832, 448, 1280, 720),
    ("16:9 (Ngang)", "Full HD (1080p - Siêu nét)"): (832, 448, 1920, 1080),
    ("9:16 (Dọc Shorts/Reels)", "SD (480p - Gốc siêu nhanh)"): (448, 832, 448, 832),
    ("9:16 (Dọc Shorts/Reels)", "HD (720p - Sắc nét)"): (448, 832, 720, 1280),
    ("9:16 (Dọc Shorts/Reels)", "Full HD (1080p - Siêu nét)"): (448, 832, 1080, 1920),
    ("1:1 (Vuông)", "SD (480p - Gốc siêu nhanh)"): (624, 624, 624, 624),
    ("1:1 (Vuông)", "HD (720p - Sắc nét)"): (624, 624, 720, 720),
    ("1:1 (Vuông)", "Full HD (1080p - Siêu nét)"): (624, 624, 1080, 1080),
}

DURATION_CHOICES = ["3s", "4s", "5s", "6s (Đề xuất tối ưu)", "8s", "10s", "12s", "18s", "20s", "25s", "30s", "Tùy chỉnh"]

# 4k + 1 aligned frames for each duration at 16 FPS
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
