"""
FastVideo AI Studio Pro - Production Web UI
===========================================
Enterprise-grade video generation studio powered by FastVideo,
Wan2.1 / Wan2.2 DiT (1.3B 480p & 5B 720p), Triton Video Sparse Attention,
and Lanczos4 High-Resolution Upscaling.
"""

import os
import time
import random
from copy import deepcopy
from typing import Dict, List, Tuple

import cv2
import gradio as gr
import torch

from fastvideo.entrypoints.video_generator import VideoGenerator
from fastvideo.api.sampling_param import SamplingParam

from studio.config import (
    MODEL_PATH_MAPPING,
    DURATION_CHOICES,
    DURATION_TO_FRAMES,
    MULTI_SCENE_DURATIONS,
    DEFAULT_FPS,
    NUM_CARDS,
    get_resolution_config,
)
from studio.postprocess import (
    upscale_video_file,
    concatenate_video_files,
    parse_multi_scenes,
)
from studio.ui_helpers import (
    create_timing_display,
    safe_progress,
    make_progress_bar_html,
    STUDIO_CSS,
)


def setup_model_environment(model_path: str) -> None:
    """Configure low-level attention backends for peak GPU performance."""
    if "fullattn" in model_path.lower():
        try:
            import flash_attn  # noqa: F401
            os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"
        except ImportError:
            os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"
    else:
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"


def get_or_load_generator(
    model_name_or_key: str,
    generators: Dict[str, VideoGenerator],
    default_params: Dict[str, SamplingParam],
    progress=None,
    step_cb=None,
) -> Tuple[VideoGenerator, SamplingParam]:
    """Retrieve preloaded generator or initialize dynamically on demand."""
    model_path = MODEL_PATH_MAPPING.get(model_name_or_key, model_name_or_key)

    if model_path in generators:
        return generators[model_path], default_params[model_path]

    msg = f"📦 Đang nạp mô hình {model_name_or_key} ({model_path}) vào GPU..."
    safe_progress(progress, 0.05, desc=msg)
    if step_cb:
        step_cb(5, msg)
    print(f"\n[Studio] Loading model dynamically: {model_path}")

    setup_model_environment(model_path)
    torch.cuda.empty_cache()

    gen = VideoGenerator.from_pretrained(
        model_path,
        num_gpus=1,
        text_encoder_cpu_offload=True,
        dit_layerwise_offload=False,
        dit_cpu_offload=True,
        vae_cpu_offload=True,
    )
    param = SamplingParam.from_pretrained(model_path)
    generators[model_path] = gen
    default_params[model_path] = param
    return gen, param


def load_studio_prompts() -> Tuple[List[str], List[str]]:
    """Load curated studio prompts from file or fallback to high-quality presets."""
    prompt_file = os.path.join(os.path.dirname(__file__), "prompts.txt")
    prompts: List[str] = []
    labels: List[str] = []

    if os.path.exists(prompt_file):
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lbl = line[:90] + "..." if len(line) > 90 else line
                        labels.append(lbl)
                        prompts.append(line)
        except Exception as e:
            print(f"Warning: Could not read {prompt_file}: {e}")

    if not prompts:
        prompts = [
            "A majestic golden eagle soaring over snowy mountain peaks during a vibrant crimson sunset, 8k resolution, cinematic lighting.",
            "A cyberpunk samurai walking through a rainy neon-lit street in Tokyo at midnight, reflections on wet pavement, cinematic depth.",
            "A cute red panda wearing a tiny knitted sweater drinking hot chocolate by a cozy wooden fireplace in a winter cabin.",
        ]
        labels = [p[:90] + "..." for p in prompts]

    return prompts, labels


def create_studio_interface(
    default_params: Dict[str, SamplingParam],
    generators: Dict[str, VideoGenerator],
) -> gr.Blocks:
    """Construct the complete Gradio Studio interface."""

    def render_single_video_flow(
        prompt: str,
        negative_prompt: str,
        use_negative_prompt: bool,
        seed: int,
        guidance_scale: float,
        num_frames: int,
        height: int,
        width: int,
        aspect_ratio: str,
        resolution: str,
        generation_mode: str,
        multi_duration: str,
        model_selection: str,
        progress=None,
        step_cb=None,
    ):
        model_path = MODEL_PATH_MAPPING.get(model_selection, model_selection)
        setup_model_environment(model_path)
        generator, param_template = get_or_load_generator(
            model_selection,
            generators,
            default_params,
            progress=progress,
            step_cb=step_cb,
        )

        params = deepcopy(param_template)
        output_dir = "outputs/"
        os.makedirs(output_dir, exist_ok=True)
        total_start_time = time.time()

        # Dynamically determine native and target dimensions
        res_cfg = get_resolution_config(model_selection, aspect_ratio, resolution)
        native_w, native_h, target_w, target_h = res_cfg

        params.guidance_scale = guidance_scale
        params.height = native_h
        params.width = native_w
        if use_negative_prompt and negative_prompt:
            params.negative_prompt = negative_prompt
        elif hasattr(param_template, "negative_prompt") and param_template.negative_prompt:
            params.negative_prompt = param_template.negative_prompt

        params.seed = int(seed)

        # ----------------------------------------------------
        # 1. Multi-Scene Extension Mode (15s - 60s)
        # ----------------------------------------------------
        if "Đa cảnh" in generation_mode:
            num_scenes_map = {
                "15s (3 phân cảnh x 5s)": 3,
                "20s (4 phân cảnh x 5s)": 4,
                "30s (6 phân cảnh x 5s)": 6,
                "60s (12 phân cảnh x 5s)": 12,
            }
            n_scenes = num_scenes_map.get(multi_duration, 4)
            scene_prompts = parse_multi_scenes(prompt, n_scenes)
            scene_paths = []
            base_seed = params.seed

            for s_idx, s_prompt in enumerate(scene_prompts):
                pct = int(10 + 70 * (s_idx / n_scenes))
                desc = f"🎬 Render Cảnh {s_idx + 1}/{n_scenes} (5s)..."
                safe_progress(progress, (s_idx / n_scenes) * 0.85 + 0.05, desc=desc)
                if step_cb:
                    step_cb(pct, desc)

                params.prompt = s_prompt
                params.seed = base_seed + s_idx * 37
                params.num_frames = 81  # 5 seconds at 16 fps
                result = generator.generate_video(
                    prompt=s_prompt,
                    sampling_param=params,
                    save_video=True,
                    return_frames=False,
                )
                scene_path = result.get("video_path")
                if not scene_path or not os.path.exists(scene_path):
                    mp4s = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".mp4")]
                    if mp4s:
                        scene_path = max(mp4s, key=os.path.getmtime)
                if scene_path:
                    scene_paths.append(scene_path)
                torch.cuda.empty_cache()

            safe_progress(progress, 0.90, desc="🔗 Đang ghép nối các phân cảnh...")
            if step_cb:
                step_cb(85, "🔗 Ghép nối các phân cảnh thành video hoàn chỉnh...")

            safe_title = "".join(c for c in prompt[:40] if c.isalnum() or c in (" ", "_", "-")).strip()
            merged_path = os.path.join(output_dir, f"Studio_{safe_title}_{n_scenes * 5}s.mp4")
            final_merged = concatenate_video_files(scene_paths, merged_path, fps=DEFAULT_FPS)

            if (target_w, target_h) != (native_w, native_h):
                safe_progress(progress, 0.95, desc=f"✨ Nâng cấp siêu phân giải sang {resolution}...")
                if step_cb:
                    step_cb(92, f"✨ Nâng cấp siêu phân giải sang {resolution} (Lanczos4 + Unsharp)...")
                upscaled_path = os.path.splitext(final_merged)[0] + f"_{target_w}x{target_h}.mp4"
                final_path = upscale_video_file(final_merged, upscaled_path, target_w, target_h, fps=DEFAULT_FPS)
            else:
                final_path = final_merged

            total_time = time.time() - total_start_time
            safe_progress(progress, 1.0, desc="Hoàn tất video đa cảnh!")
            if step_cb:
                step_cb(100, f"Hoàn tất ({total_time:.1f}s)!")
            return os.path.abspath(final_path), params.seed, total_time, target_w, target_h

        # ----------------------------------------------------
        # 2. Single-Shot Mode
        # ----------------------------------------------------
        else:
            safe_progress(progress, 0.2, desc="Chuẩn bị tham số...")
            if step_cb:
                step_cb(20, "Chuẩn bị tham số...")

            params.prompt = prompt
            nf = int(num_frames)
            if (nf - 1) % 4 != 0:
                nf = ((nf - 1 + 2) // 4) * 4 + 1
            params.num_frames = nf

            safe_progress(progress, 0.4, desc="Đang sinh video trên GPU...")
            if step_cb:
                step_cb(40, f"Đang sinh video trên GPU ({nf} frames)...")

            result = generator.generate_video(
                prompt=prompt,
                sampling_param=params,
                save_video=True,
                return_frames=False,
            )
            total_time = time.time() - total_start_time

            output_path = result.get("video_path")
            if not output_path or not os.path.exists(output_path):
                safe_prompt = params.prompt[:100].strip().strip(".")
                candidate = os.path.join(output_dir, f"{safe_prompt}.mp4")
                if os.path.exists(candidate):
                    output_path = candidate
                else:
                    mp4s = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".mp4")]
                    if mp4s:
                        output_path = max(mp4s, key=os.path.getmtime)

            if output_path and (target_w, target_h) != (native_w, native_h):
                safe_progress(progress, 0.92, desc=f"✨ Nâng cấp siêu phân giải sang {resolution}...")
                if step_cb:
                    step_cb(88, f"✨ Nâng cấp siêu phân giải sang {resolution} (Lanczos4 + Unsharp)...")
                upscaled_path = os.path.splitext(output_path)[0] + f"_{target_w}x{target_h}.mp4"
                output_path = upscale_video_file(output_path, upscaled_path, target_w, target_h, fps=DEFAULT_FPS)

            if output_path:
                output_path = os.path.abspath(output_path)

            safe_progress(progress, 1.0, desc="Hoàn tất video!")
            if step_cb:
                step_cb(100, f"Hoàn tất ({total_time:.1f}s)!")

            return output_path, params.seed, total_time, target_w, target_h

    examples, example_labels = load_studio_prompts()

    theme = gr.themes.Soft(
        primary_hue="rose",
        neutral_hue="slate",
    ).set(
        body_background_fill="#0a0d14",
        body_background_fill_dark="#0a0d14",
        block_background_fill="transparent",
        block_background_fill_dark="transparent",
        block_border_color="transparent",
        block_border_color_dark="transparent",
        block_border_width="0px",
        block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        block_label_border_color="transparent",
        block_label_border_color_dark="transparent",
        block_label_border_width="0px",
        block_label_text_color="#8b9bb4",
        block_label_text_color_dark="#8b9bb4",
        input_background_fill="#0b0f15",
        input_background_fill_dark="#0b0f15",
        input_border_color="#243142",
        input_border_color_dark="#243142",
        input_border_width="1px",
        button_primary_background_fill="#e63946",
        button_primary_background_fill_dark="#e63946",
        button_primary_background_fill_hover="#ff4d5e",
        button_primary_text_color="#ffffff",
        body_text_color="#f0f6fc",
        body_text_color_dark="#f0f6fc",
    )

    available_models = [
        "FastWan2.1-T2V-1.3B (480p - Siêu tốc)",
        "FastWan2.2-TI2V-5B (720p - Sparse VSA)",
        "FastWan2.2-TI2V-5B-FullAttn (720p - Flash/SDPA)",
    ]

    with gr.Blocks(
        title="FastVideo AI Studio Pro",
        theme=theme,
        css=STUDIO_CSS,
        js="() => { document.documentElement.classList.add('dark'); document.body.classList.add('dark'); }",
    ) as studio_app:
        # Hero Studio Header Bar
        with gr.Row(elem_classes="studio-hero-header"):
            gr.HTML("""
            <div style="display: flex; align-items: center; justify-content: space-between; width: 100%; flex-wrap: wrap; gap: 10px;">
                <div class="hero-title-group">
                    <div style="display: flex; align-items: center; justify-content: center; width: 42px; height: 42px; background: linear-gradient(135deg, #e63946, #ff4d5e); border-radius: 9px; box-shadow: 0 4px 14px rgba(230, 57, 70, 0.45); flex-shrink: 0;">
                        <span style="font-size: 20px;">🎬</span>
                    </div>
                    <div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 18px; font-weight: 800; color: #f0f6fc; letter-spacing: -0.4px;">FastVideo AI Studio Pro</span>
                            <span style="font-size: 10px; font-weight: 700; background: rgba(245, 166, 35, 0.2); color: #ffbe4d; border: 1px solid rgba(245, 166, 35, 0.4); padding: 1px 7px; border-radius: 10px;">PRO v2.5</span>
                        </div>
                        <p style="margin: 2px 0 0 0; color: #8b9bb4; font-size: 11px;">Studio Sản Xuất Video AI Thương Mại • Wan2.1 Diffusion • Triton VSA Native • Siêu Phân Giải Lanczos4</p>
                    </div>
                </div>
                <div class="hero-badges-group">
                    <span class="hero-badge badge-active"><span style="color: #10b981; font-size: 9px;">●</span> RTX 5060 Ti (16GB)</span>
                    <span class="hero-badge badge-highlight">⚡ Triton VSA 50x</span>
                    <span class="hero-badge">📦 FastWan 2.1 & 2.2</span>
                </div>
            </div>
            """)

        # Main 2-Column Layout
        with gr.Row(equal_height=False, elem_classes="main-studio-row"):
            # ========================================================
            # CỘT TRÁI (LEFT COLUMN): CẤU HÌNH & NHẬP LIỆU
            # ========================================================
            with gr.Column(scale=5, elem_classes="studio-col-left"):
                # Card 1: Kịch bản & Mô hình
                with gr.Group(elem_classes="studio-card"):
                    gr.HTML("<div class='card-title'><span>📝</span> KỊCH BẢN & MÔ HÌNH SẢN XUẤT</div>")

                    with gr.Row():
                        model_selection = gr.Dropdown(
                            choices=available_models,
                            value=available_models[0],
                            label="🤖 Chọn Mô Hình (Model)",
                            scale=3,
                            interactive=True,
                        )
                        example_dropdown = gr.Dropdown(
                            choices=example_labels,
                            label="💡 Mẫu Prompt Điện Ảnh (Presets)",
                            value=None,
                            scale=2,
                            interactive=True,
                            allow_custom_value=False,
                        )

                    prompt_mode = gr.Radio(
                        choices=["🎯 Tạo Đơn (Single)", "📋 Hàng Đợi (Mỗi Dòng 1 Video)"],
                        value="🎯 Tạo Đơn (Single)",
                        label="Chế Độ Kịch Bản",
                        interactive=True,
                    )

                    prompt = gr.Textbox(
                        label="Kịch Bản Prompt (Text-to-Video)",
                        placeholder="Nhập mô tả video bạn muốn tạo (Ví dụ: A majestic golden eagle soaring over snowy mountain peaks at sunset, cinematic 4k, photorealistic)...",
                        lines=5,
                        max_lines=14,
                        autofocus=True,
                    )

                    prompt_mode_hint = gr.Markdown(
                        value="💡 **Chế độ Tạo Đơn**: Cả khung kịch bản trên sẽ được dùng để tạo ra **1 video duy nhất**.",
                        visible=True,
                    )

                    with gr.Accordion("🚫 Negative Prompt (Lọc Chi Tiết Xấu / Artifacts)", open=False):
                        use_negative_prompt = gr.Checkbox(label="Kích hoạt bộ lọc Negative Prompt", value=False)
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="low quality, blurry, distorted, artifacts, watermark, worst quality, deformed...",
                            lines=2,
                            visible=False,
                        )

                # Card 2: Thiết lập video
                with gr.Group(elem_classes="studio-card"):
                    gr.HTML("<div class='card-title'><span>⚙️</span> THIẾT LẬP ĐỊNH DẠNG & ĐỘ PHÂN GIẢI</div>")

                    with gr.Row():
                        aspect_ratio = gr.Dropdown(
                            choices=["16:9 (Ngang - YouTube)", "9:16 (Dọc - Shorts/TikTok)", "1:1 (Vuông - Feed)"],
                            value="16:9 (Ngang - YouTube)",
                            label="📐 Tỷ Lệ Khung Hình",
                            interactive=True,
                        )
                        resolution = gr.Dropdown(
                            choices=["SD (480p - Gốc Siêu Tốc)", "HD (720p - Sắc Nét)", "Full HD (1080p - Siêu Nét)"],
                            value="SD (480p - Gốc Siêu Tốc)",
                            label="📺 Độ Phân Giải Xuất",
                            interactive=True,
                        )

                    resolution_info = gr.Markdown(
                        value="📐 Độ phân giải xuất: **832 x 448** (Render gốc 16:9 siêu tốc Wan2.1)",
                        visible=True,
                    )

                    with gr.Row():
                        generation_mode = gr.Dropdown(
                            choices=["Đơn Cảnh (Single Shot)", "Nối Tiếp Đa Cảnh (Multi-Scene 15s-60s)"],
                            value="Đơn Cảnh (Single Shot)",
                            label="🎬 Chế Độ Quay",
                            interactive=True,
                        )

                    with gr.Row(visible=True) as single_duration_row:
                        duration_dropdown = gr.Dropdown(
                            label="⏱️ Thời Lượng Video (Đơn Cảnh)",
                            choices=DURATION_CHOICES,
                            value="6s (Đề xuất tối ưu)",
                            interactive=True,
                        )

                    with gr.Row(visible=False) as multi_duration_row:
                        multi_duration = gr.Dropdown(
                            label="⏱️ Tổng Thời Lượng Đa Cảnh (15s - 60s)",
                            choices=MULTI_SCENE_DURATIONS,
                            value="20s (4 phân cảnh x 5s)",
                            interactive=True,
                        )

                    duration_note = gr.Markdown(
                        value="✅ **Tối ưu**: Mốc **6s (97 frames @ 16 FPS)** an toàn và tối ưu cho RTX 5060 Ti.",
                        visible=True,
                    )

                # Card 3: Thông số nâng cao
                with gr.Accordion("🔧 THÔNG SỐ NÂNG CAO (CFG, SEED, FRAMES)", open=False):
                    with gr.Row():
                        guidance_scale = gr.Slider(
                            label="Guidance Scale (CFG)",
                            minimum=1.0,
                            maximum=12.0,
                            step=0.5,
                            value=3.0,
                        )
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=1000000,
                            step=1,
                            value=1024,
                        )
                    with gr.Row():
                        randomize_seed = gr.Checkbox(label="🎲 Randomize seed", value=False)
                        num_frames = gr.Number(
                            label="Số Khung Hình (Frames = 4k + 1)",
                            value=97,
                            interactive=True,
                        )
                    with gr.Row():
                        height = gr.Number(label="Chiều Cao (Height)", value=448, interactive=True)
                        width = gr.Number(label="Chiều Rộng (Width)", value=832, interactive=True)

                # Nút Run lớn, nổi bật
                run_button = gr.Button(
                    "🚀 BẮT ĐẦU TẠO VIDEO (GENERATE)",
                    variant="primary",
                    size="lg",
                    elem_classes="btn-generate-main",
                )

            # ========================================================
            # CỘT PHẢI (RIGHT COLUMN): LIST VIDEO & TIẾN TRÌNH DẠNG DÒNG
            # ========================================================
            with gr.Column(scale=7, elem_classes="studio-col-right"):
                gr.HTML("""
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                    <div style="font-size: 13px; font-weight: 700; color: #f0f6fc; text-transform: uppercase; letter-spacing: 0.5px;">🎬 HÀNG ĐỢI & KẾT QUẢ VIDEO</div>
                    <span style="font-size: 11px; color: #8b9bb4; background: #0b0f15; border: 1px solid #243142; padding: 3px 10px; border-radius: 12px;">Tự động cập nhật trực tiếp</span>
                </div>
                """)

                overall_status = gr.HTML(
                    value="<div class='banner-info'><span>⚡</span> <span><strong>Hệ thống sẵn sàng:</strong> Nhập prompt bên trái và nhấn <strong>BẮT ĐẦU TẠO VIDEO</strong> để khởi chạy.</span></div>",
                    visible=True,
                )

                card_groups = []
                card_headers = []
                card_statuses = []
                card_progresses = []
                card_videos = []
                card_metas = []

                for i in range(NUM_CARDS):
                    is_first = (i == 0)
                    with gr.Group(visible=is_first, elem_classes="video-row-card") as c_grp:
                        with gr.Row(elem_classes="card-header-row"):
                            c_hdr = gr.Markdown(f"### 🎬 Phân Cảnh #{i+1}")
                            c_sta = gr.HTML("<span class='status-badge badge-idle'>Chờ Lệnh</span>")
                        c_prg = gr.HTML(make_progress_bar_html(0, "Chờ bắt đầu...", "idle"))
                        c_vid = gr.Video(show_label=False, height=380, interactive=False)
                        c_met = gr.Markdown(value="", visible=False)

                    card_groups.append(c_grp)
                    card_headers.append(c_hdr)
                    card_statuses.append(c_sta)
                    card_progresses.append(c_prg)
                    card_videos.append(c_vid)
                    card_metas.append(c_met)

                # Video Extension Panel
                with gr.Accordion("➕ NỐI TIẾP THÊM +5S VÀO VIDEO ĐÃ TẠO (VIDEO EXTENSION)", open=False):
                    extend_target = gr.Dropdown(
                        choices=[f"Video #{i+1}" for i in range(NUM_CARDS)],
                        value="Video #1",
                        label="🎯 Chọn Video Muốn Nối Tiếp",
                    )
                    extend_prompt = gr.Textbox(
                        label="Prompt Phân Cảnh Nối Tiếp (+5s)",
                        placeholder="Mô tả hành động tiếp theo (để trống sẽ tự động tiếp diễn theo ngữ cảnh)...",
                        lines=2,
                    )
                    extend_button = gr.Button("🎬 Nối Tiếp Phân Cảnh (+5s)", variant="secondary")
                    extend_status = gr.Markdown(visible=False)

        # ----------------------------------------------------
        # EVENT BINDINGS
        # ----------------------------------------------------
        def on_example_select(example_label: str) -> str:
            if example_label and example_label in example_labels:
                idx = example_labels.index(example_label)
                return examples[idx]
            return ""

        example_dropdown.change(
            fn=on_example_select,
            inputs=example_dropdown,
            outputs=prompt,
        )

        def on_prompt_mode_change(mode: str):
            if "đơn" in mode.lower() or "single" in mode.lower():
                placeholder = "Nhập mô tả video bạn muốn tạo (Ví dụ: A majestic golden eagle soaring over snowy mountain peaks at sunset, cinematic 4k, photorealistic)..."
                hint = "💡 **Chế độ Tạo Đơn**: Cả khung kịch bản trên sẽ được dùng để tạo ra **1 video duy nhất**."
            else:
                placeholder = (
                    "Nhập mỗi dòng là 1 prompt riêng biệt để tạo nhiều video liên tiếp:\n"
                    "Dòng 1: Chú mèo đội nón lá uống cà phê bên bờ hồ Hoàn Kiếm\n"
                    "Dòng 2: Phi thuyền không gian bay qua các hành tinh lấp lánh ánh sao\n"
                    "Dòng 3: Khung cảnh hoàng hôn rực rỡ trên bãi biển nhiệt đới..."
                )
                hint = "💡 **Chế độ Hàng Đợi**: Mỗi dòng sẽ được tạo thành **1 video riêng biệt** tuần tự trên GPU."
            return gr.update(placeholder=placeholder), gr.update(value=hint)

        prompt_mode.change(
            fn=on_prompt_mode_change,
            inputs=prompt_mode,
            outputs=[prompt, prompt_mode_hint],
        )

        use_negative_prompt.change(
            fn=lambda x: gr.update(visible=x),
            inputs=use_negative_prompt,
            outputs=negative_prompt,
        )

        def on_duration_change(choice: str):
            if choice in DURATION_TO_FRAMES:
                frames = DURATION_TO_FRAMES[choice]
                if choice in ["12s", "18s", "20s", "25s", "30s"]:
                    msg = f"⚠️ **Cảnh báo**: Mốc **{choice} ({frames} frames)** đòi hỏi nhiều VRAM. Khuyến nghị dùng chế độ **Nối tiếp đa cảnh** để đạt chất lượng tốt nhất."
                elif choice in ["8s", "10s"]:
                    msg = f"ℹ️ **Lưu ý**: Mốc **{choice} ({frames} frames)** tiêu tốn ~11GB-14GB VRAM. Hoạt động tốt trên RTX 5060 Ti."
                else:
                    msg = f"✅ **Tối ưu**: Mốc **{choice} ({frames} frames @ 16 FPS)** an toàn và tối ưu cho RTX 5060 Ti."
                return gr.update(value=frames), gr.update(value=msg, visible=True)
            elif choice == "Tùy chỉnh":
                return gr.update(), gr.update(value="✍️ Nhập số frame tùy ý trong mục Thông số nâng cao (tự động làm tròn 4k + 1).", visible=True)
            return gr.update(), gr.update(visible=False)

        duration_dropdown.change(
            fn=on_duration_change,
            inputs=duration_dropdown,
            outputs=[num_frames, duration_note],
        )

        def on_ratio_or_res_change(model_sel: str, ratio: str, res: str):
            cfg = get_resolution_config(model_sel, ratio, res)
            native_w, native_h, target_w, target_h = cfg
            is_5b = ("5B" in model_sel or "720p" in model_sel.lower() or "5b" in model_sel.lower())

            if is_5b:
                if (target_w, target_h) == (native_w, native_h):
                    msg = f"📐 **Mô hình 5B Native 720P**: Render trực tiếp **{target_w} x {target_h}** sắc nét chuẩn điện ảnh!"
                else:
                    msg = f"📐 **Mô hình 5B Native 720P**: Render gốc {native_w}x{native_h} -> Siêu phân giải Lanczos4 lên **{target_w} x {target_h}**!"
            else:
                if (target_w, target_h) == (native_w, native_h):
                    msg = f"📐 **Mô hình 1.3B**: Render gốc siêu tốc **{target_w} x {target_h}** (Wan2.1 480P)"
                else:
                    msg = f"📐 **Mô hình 1.3B**: Render gốc {native_w}x{native_h} -> Siêu phân giải Lanczos4 lên **{target_w} x {target_h}**!"
            return gr.update(value=target_h), gr.update(value=target_w), gr.update(value=msg)

        aspect_ratio.change(
            fn=on_ratio_or_res_change,
            inputs=[model_selection, aspect_ratio, resolution],
            outputs=[height, width, resolution_info],
        )

        resolution.change(
            fn=on_ratio_or_res_change,
            inputs=[model_selection, aspect_ratio, resolution],
            outputs=[height, width, resolution_info],
        )

        def on_model_change(model_sel: str, ratio: str):
            is_5b = ("5B" in model_sel or "720p" in model_sel.lower() or "5b" in model_sel.lower())
            if is_5b:
                res_val = "HD (720p - Sắc nét)"
            else:
                res_val = "SD (480p - Gốc siêu nhanh)"

            cfg = get_resolution_config(model_sel, ratio, res_val)
            native_w, native_h, target_w, target_h = cfg
            if is_5b:
                msg = f"📐 **Mô hình 5B Native 720P**: Render trực tiếp **{target_w} x {target_h}** sắc nét chuẩn điện ảnh!"
            else:
                msg = f"📐 **Mô hình 1.3B**: Render gốc siêu tốc **{target_w} x {target_h}** (Wan2.1 480P)"
            return gr.update(value=res_val), gr.update(value=target_h), gr.update(value=target_w), gr.update(value=msg)

        model_selection.change(
            fn=on_model_change,
            inputs=[model_selection, aspect_ratio],
            outputs=[resolution, height, width, resolution_info],
        )

        def on_mode_change(mode: str):
            if "Đa cảnh" in mode:
                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(value="🎬 **Chế độ Đa Cảnh Studio**: Render tuần tự từng cảnh 5s và ghép mượt mà không lo tràn VRAM!"),
                )
            else:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(value="✅ **Chế độ Đơn Cảnh**: Tạo 1 phân cảnh duy nhất (3s - 10s)."),
                )

        generation_mode.change(
            fn=on_mode_change,
            inputs=generation_mode,
            outputs=[single_duration_row, multi_duration_row, duration_note],
        )

        # ----------------------------------------------------
        # BATCH QUEUE GENERATOR (LIVE YIELD)
        # ----------------------------------------------------
        def handle_studio_generation(
            model_sel,
            p_mode,
            raw_prompt,
            neg_prompt,
            use_neg,
            s_val,
            gs_val,
            nf_val,
            h_val,
            w_val,
            rand_seed,
            ratio_val,
            res_val,
            gen_mode,
            multi_dur,
            progress=gr.Progress(),
        ):
            if "Tạo đơn" in p_mode:
                prompts_to_run = [raw_prompt.strip()] if raw_prompt and raw_prompt.strip() else []
            else:
                prompts_to_run = [line.strip() for line in raw_prompt.splitlines() if line.strip()]

            if not prompts_to_run:
                yield (
                    "<div class='banner-error'>⚠️ Vui lòng nhập ít nhất 1 dòng prompt!</div>",
                    *[gr.update() for _ in range(NUM_CARDS * 6)],
                    gr.update(),
                )
                return

            num_runs = min(len(prompts_to_run), NUM_CARDS)
            base_seed = random.randint(0, 1000000) if rand_seed else int(s_val)

            card_updates = []
            for i in range(NUM_CARDS):
                if i < num_runs:
                    p_snip = prompts_to_run[i][:60] + ("..." if len(prompts_to_run[i]) > 60 else "")
                    card_updates.extend([
                        gr.update(visible=True),
                        gr.update(value=f"### 🎬 Video #{i+1}: *{p_snip}*"),
                        "<span class='status-badge badge-idle'>⏳ Chờ trong hàng đợi</span>",
                        make_progress_bar_html(0, "Chờ bắt đầu...", "idle"),
                        gr.update(value=None),
                        gr.update(visible=False, value=""),
                    ])
                else:
                    card_updates.extend([
                        gr.update(visible=False),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    ])

            init_banner = f"<div class='banner-info'>📋 Hàng đợi: <strong>{num_runs} video</strong> cần tạo với mô hình <strong>{model_sel}</strong>. Bắt đầu xử lý tuần tự...</div>"
            ext_choices = [f"Video #{k+1}" for k in range(num_runs)]
            yield (init_banner, *card_updates, gr.update(choices=ext_choices, value=ext_choices[0]))

            active_cards = {}
            for i in range(NUM_CARDS):
                if i < num_runs:
                    p_snip = prompts_to_run[i][:60] + ("..." if len(prompts_to_run[i]) > 60 else "")
                    active_cards[i] = {
                        "visible": True,
                        "header": f"### 🎬 Video #{i+1}: *{p_snip}*",
                        "status": "<span class='status-badge badge-idle'>⏳ Chờ trong hàng đợi</span>",
                        "progress": make_progress_bar_html(0, "Chờ bắt đầu...", "idle"),
                        "video": None,
                        "meta": gr.update(visible=False, value=""),
                    }
                else:
                    active_cards[i] = {
                        "visible": False,
                        "header": gr.update(),
                        "status": gr.update(),
                        "progress": gr.update(),
                        "video": gr.update(),
                        "meta": gr.update(),
                    }

            def build_yield_pack(banner_html):
                pack = [banner_html]
                for i in range(NUM_CARDS):
                    c = active_cards[i]
                    pack.extend([
                        gr.update(visible=c["visible"]) if isinstance(c["visible"], bool) else c["visible"],
                        c["header"],
                        c["status"],
                        c["progress"],
                        c["video"],
                        c["meta"],
                    ])
                pack.append(gr.update(choices=ext_choices, value=ext_choices[0]))
                return tuple(pack)

            for idx in range(num_runs):
                cur_prompt = prompts_to_run[idx]
                cur_seed = base_seed + idx * 79

                active_cards[idx]["status"] = "<span class='status-badge badge-running'>⚡ Đang xử lý GPU...</span>"
                active_cards[idx]["progress"] = make_progress_bar_html(10, "Đang khởi tạo...", "running")
                banner_now = f"<div class='banner-info'>⚡ Đang xử lý <strong>Video #{idx+1}/{num_runs}</strong> ({model_sel}): <em>{cur_prompt[:60]}...</em></div>"
                safe_progress(progress, (idx / num_runs), desc=f"Video #{idx+1}/{num_runs}: Đang tạo...")
                yield build_yield_pack(banner_now)

                def make_step_cb(c_idx):
                    def step_callback(pct, desc):
                        active_cards[c_idx]["progress"] = make_progress_bar_html(pct, desc, "running")
                    return step_callback

                try:
                    out_path, used_seed, item_time, target_w, target_h = render_single_video_flow(
                        prompt=cur_prompt,
                        negative_prompt=neg_prompt,
                        use_negative_prompt=use_neg,
                        seed=cur_seed,
                        guidance_scale=gs_val,
                        num_frames=nf_val,
                        height=h_val,
                        width=w_val,
                        aspect_ratio=ratio_val,
                        resolution=res_val,
                        generation_mode=gen_mode,
                        multi_duration=multi_dur,
                        model_selection=model_sel,
                        progress=progress,
                        step_cb=make_step_cb(idx),
                    )

                    active_cards[idx]["status"] = "<span class='status-badge badge-success'>✅ Hoàn thành</span>"
                    active_cards[idx]["progress"] = make_progress_bar_html(100, f"Hoàn tất trong {item_time:.1f}s!", "success")
                    active_cards[idx]["video"] = out_path
                    active_cards[idx]["meta"] = gr.update(
                        visible=True,
                        value=f"⏱️ Thời gian: **{item_time:.1f}s** | 📐 Phân giải: **{target_w}x{target_h}** | 🌱 Seed: `{used_seed}`",
                    )
                    torch.cuda.empty_cache()

                except Exception as ex:
                    import traceback
                    traceback.print_exc()
                    active_cards[idx]["status"] = "<span class='status-badge badge-error'>❌ Thất bại</span>"
                    active_cards[idx]["progress"] = make_progress_bar_html(100, f"Lỗi: {str(ex)}", "error")
                    active_cards[idx]["meta"] = gr.update(visible=True, value=f"⚠️ Lỗi chi tiết: `{str(ex)}`")

                safe_progress(progress, (idx + 1) / num_runs, desc=f"Video #{idx+1} hoàn thành.")
                yield build_yield_pack(banner_now)

            final_banner = f"<div class='banner-success'>🎉 Đã hoàn tất toàn bộ <strong>{num_runs} video</strong> trong hàng đợi!</div>"
            safe_progress(progress, 1.0, desc="Đã hoàn tất toàn bộ hàng đợi!")
            yield build_yield_pack(final_banner)

        all_card_outputs = []
        for i in range(NUM_CARDS):
            all_card_outputs.extend([
                card_groups[i],
                card_headers[i],
                card_statuses[i],
                card_progresses[i],
                card_videos[i],
                card_metas[i],
            ])

        run_button.click(
            fn=handle_studio_generation,
            inputs=[
                model_selection,
                prompt_mode,
                prompt,
                negative_prompt,
                use_negative_prompt,
                seed,
                guidance_scale,
                num_frames,
                height,
                width,
                randomize_seed,
                aspect_ratio,
                resolution,
                generation_mode,
                multi_duration,
            ],
            outputs=[overall_status, *all_card_outputs, extend_target],
            concurrency_limit=20,
        )

        # ----------------------------------------------------
        # EXTEND ACTION HANDLER (+5s)
        # ----------------------------------------------------
        def handle_extend_action(
            chosen_target,
            ext_prompt,
            model_sel,
            neg_prompt,
            use_neg,
            s,
            gs,
            ratio,
            res,
            *vid_list,
            progress=gr.Progress(),
        ):
            try:
                target_idx = int(chosen_target.replace("Video #", "").strip()) - 1
            except Exception:
                target_idx = 0

            if target_idx < 0 or target_idx >= len(vid_list):
                return gr.update(visible=True, value="⚠️ Không tìm thấy vị trí video!"), *[gr.update() for _ in range(NUM_CARDS)]

            current_video = vid_list[target_idx]
            if not current_video or not os.path.exists(current_video):
                return gr.update(visible=True, value=f"⚠️ {chosen_target} chưa có video để nối tiếp! Hãy tạo video trước."), *[gr.update() for _ in range(NUM_CARDS)]

            model_path = MODEL_PATH_MAPPING.get(model_sel, model_sel)
            generator, param_template = get_or_load_generator(model_sel, generators, default_params, progress=progress)
            params = deepcopy(param_template)

            res_cfg = get_resolution_config(model_sel, ratio, res)
            native_w, native_h, target_w, target_h = res_cfg

            prompt_to_use = ext_prompt.strip() if ext_prompt and ext_prompt.strip() else "continuous scene, cinematic lighting, fluid action"
            params.prompt = prompt_to_use
            params.seed = int(s) + random.randint(100, 99999)
            params.guidance_scale = gs
            params.num_frames = 81  # 5s
            params.height = native_h
            params.width = native_w
            if use_neg and neg_prompt:
                params.negative_prompt = neg_prompt

            safe_progress(progress, 0.3, desc="🎬 Đang render phân cảnh nối tiếp (+5s)...")
            result_ext = generator.generate_video(
                prompt=prompt_to_use,
                sampling_param=params,
                save_video=True,
                return_frames=False,
            )
            new_segment = result_ext.get("video_path")
            if not new_segment or not os.path.exists(new_segment):
                mp4s = [os.path.join("outputs", f) for f in os.listdir("outputs") if f.endswith(".mp4")]
                if mp4s:
                    new_segment = max(mp4s, key=os.path.getmtime)

            safe_progress(progress, 0.85, desc="🔗 Ghép nối vào video hiện tại...")
            output_dir = "outputs/"
            extended_file = os.path.join(output_dir, f"Extended_{int(time.time())}.mp4")
            concat_path = concatenate_video_files([current_video, new_segment], extended_file, fps=DEFAULT_FPS)

            if (target_w, target_h) != (native_w, native_h):
                safe_progress(progress, 0.95, desc=f"✨ Nâng cấp sang {res}...")
                upscaled_path = os.path.splitext(concat_path)[0] + f"_{target_w}x{target_h}.mp4"
                final_path = upscale_video_file(concat_path, upscaled_path, target_w, target_h, fps=DEFAULT_FPS)
            else:
                final_path = concat_path

            torch.cuda.empty_cache()
            safe_progress(progress, 1.0, desc="Nối tiếp thành công!")

            vid_updates = [gr.update() for _ in range(NUM_CARDS)]
            vid_updates[target_idx] = os.path.abspath(final_path)

            return gr.update(visible=True, value=f"✅ **Đã nối tiếp thành công thêm +5s vào {chosen_target}!**"), *vid_updates

        extend_button.click(
            fn=handle_extend_action,
            inputs=[
                extend_target,
                extend_prompt,
                model_selection,
                negative_prompt,
                use_negative_prompt,
                seed,
                guidance_scale,
                aspect_ratio,
                resolution,
                *card_videos,
            ],
            outputs=[extend_status, *card_videos],
        )

    return studio_app
