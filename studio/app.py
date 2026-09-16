"""
FastVideo AI Studio Pro - Production Web UI
===========================================
Enterprise-grade video generation studio powered by FastVideo, Wan2.1 DiT,
Triton Video Sparse Attention, and Lanczos4 High-Resolution Upscaling.
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
    RESOLUTION_CONFIGS,
    MULTI_SCENE_DURATIONS,
    DEFAULT_FPS,
    NUM_CARDS,
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
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"
    else:
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"


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
        model_path = MODEL_PATH_MAPPING.get(model_selection, "FastVideo/FastWan2.1-T2V-1.3B-Diffusers")
        setup_model_environment(model_path)
        generator = generators[model_path]
        params = deepcopy(default_params[model_path])
        output_dir = "outputs/"
        os.makedirs(output_dir, exist_ok=True)
        total_start_time = time.time()

        res_cfg = RESOLUTION_CONFIGS.get((aspect_ratio, resolution), (832, 448, 832, 448))
        native_w, native_h, target_w, target_h = res_cfg

        params.guidance_scale = guidance_scale
        params.height = native_h
        params.width = native_w
        if use_negative_prompt and negative_prompt:
            params.negative_prompt = negative_prompt
        else:
            params.negative_prompt = default_params[model_path].negative_prompt

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

            start_time = time.time()
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

    # Load prompt database
    examples, example_labels = load_studio_prompts()

    # Dark Obsidian Pro Theme
    theme = gr.themes.Base().set(
        body_background_fill="#0f141c",
        block_background_fill="#161f2c",
        border_color_primary="#243142",
        button_primary_background_fill="#e63946",
        button_primary_background_fill_hover="#ff4d5e",
        button_primary_text_color="#ffffff",
        slider_color="#f5a623",
        checkbox_background_color_selected="#e63946",
    )

    def get_default_values(model_name: str) -> dict:
        model_path = MODEL_PATH_MAPPING.get(model_name)
        if model_path and model_path in default_params:
            p = default_params[model_path]
            return {
                "height": p.height,
                "width": p.width,
                "num_frames": 97,
                "guidance_scale": p.guidance_scale,
                "seed": p.seed,
            }
        return {
            "height": 448,
            "width": 832,
            "num_frames": 97,
            "guidance_scale": 3.0,
            "seed": 1024,
        }

    initial_values = get_default_values("FastWan2.1-T2V-1.3B")

    with gr.Blocks(title="FastVideo AI Studio Pro", theme=theme, css=STUDIO_CSS) as studio_app:
        # Header Row
        with gr.Row():
            with gr.Column(scale=1):
                if os.path.exists("assets/full.svg"):
                    gr.Image("assets/full.svg", show_label=False, container=False, height=60)
            with gr.Column(scale=5):
                gr.HTML("""
                <div style="display: flex; flex-direction: column; justify-content: center; height: 100%;">
                    <h2 style="margin: 0; color: #f0f6fc; font-size: 24px; font-weight: 700; letter-spacing: -0.5px;">FastVideo AI Studio Pro</h2>
                    <p style="margin: 3px 0 0 0; color: #8b9bb4; font-size: 13px;">Hệ thống sản xuất video AI chuyên nghiệp • Wan2.1 Transformer • Triton Sparse Attention • Siêu phân giải Lanczos4</p>
                </div>
                """)

        # Main 2-Column Layout
        with gr.Row(equal_height=False, elem_classes="main-studio-row"):
            # ========================================================
            # CỘT TRÁI (LEFT COLUMN): CẤU HÌNH & NHẬP LIỆU
            # ========================================================
            with gr.Column(scale=5, elem_classes="studio-col-left"):
                with gr.Group(elem_classes="studio-card"):
                    gr.HTML("<div class='card-title'>📝 Nhập Prompt & Mô Hình</div>")

                    with gr.Row():
                        model_selection = gr.Dropdown(
                            choices=list(MODEL_PATH_MAPPING.keys()),
                            value="FastWan2.1-T2V-1.3B",
                            label="🤖 Chọn mô hình (Model)",
                            interactive=True,
                        )
                        example_dropdown = gr.Dropdown(
                            choices=example_labels,
                            label="💡 Mẫu prompt (Presets)",
                            value=None,
                            interactive=True,
                            allow_custom_value=False,
                        )

                    prompt_mode = gr.Radio(
                        choices=["🎯 Tạo đơn (Single Video)", "📋 Tạo nhiều video (Hàng đợi - Mỗi dòng 1 Video)"],
                        value="🎯 Tạo đơn (Single Video)",
                        label="📝 Chế độ tạo Prompt",
                        interactive=True,
                    )

                    prompt = gr.Textbox(
                        label="Nội dung Prompt",
                        placeholder="Nhập mô tả video bạn muốn tạo (Ví dụ: A majestic eagle soaring over snowy mountains at sunset, cinematic 4k)...",
                        lines=8,
                        max_lines=18,
                        autofocus=True,
                    )

                    prompt_mode_hint = gr.Markdown(
                        value="💡 **Chế độ Tạo đơn**: Cả khung văn bản trên sẽ được dùng để tạo ra **1 video duy nhất**.",
                        visible=True,
                    )

                    with gr.Accordion("🚫 Negative Prompt (Tùy chọn loại bỏ chi tiết xấu)", open=False):
                        use_negative_prompt = gr.Checkbox(label="Bật Negative Prompt", value=False)
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="low quality, blurry, distorted, artifacts, watermark...",
                            lines=2,
                            visible=False,
                        )

                # Card: Cấu hình Video
                with gr.Group(elem_classes="studio-card"):
                    gr.HTML("<div class='card-title'>⚙️ Cấu hình Video (Options)</div>")

                    with gr.Row():
                        aspect_ratio = gr.Dropdown(
                            choices=["16:9 (Ngang)", "9:16 (Dọc Shorts/Reels)", "1:1 (Vuông)"],
                            value="16:9 (Ngang)",
                            label="📐 Tỷ lệ khung hình (Aspect Ratio)",
                            interactive=True,
                        )
                        resolution = gr.Dropdown(
                            choices=["SD (480p - Gốc siêu nhanh)", "HD (720p - Sắc nét)", "Full HD (1080p - Siêu nét)"],
                            value="SD (480p - Gốc siêu nhanh)",
                            label="📺 Độ phân giải xuất ra (Resolution)",
                            interactive=True,
                        )

                    resolution_info = gr.Markdown(
                        value="📐 Độ phân giải xuất: **832 x 448** (Render gốc 16:9 siêu tốc Wan2.1)",
                        visible=True,
                    )

                    with gr.Row():
                        generation_mode = gr.Dropdown(
                            choices=["Đơn cảnh (Single Shot)", "Nối tiếp đa cảnh (Multi-Scene Studio: 15s - 60s)"],
                            value="Đơn cảnh (Single Shot)",
                            label="🎬 Chế độ tạo video (Generation Mode)",
                            interactive=True,
                        )

                    with gr.Row(visible=True) as single_duration_row:
                        duration_dropdown = gr.Dropdown(
                            label="⏱️ Thời lượng video (Đơn cảnh)",
                            choices=DURATION_CHOICES,
                            value="6s",
                            interactive=True,
                        )

                    with gr.Row(visible=False) as multi_duration_row:
                        multi_duration = gr.Dropdown(
                            label="⏱️ Tổng thời lượng đa cảnh (Multi-Scene Total Duration)",
                            choices=MULTI_SCENE_DURATIONS,
                            value="20s (4 phân cảnh x 5s)",
                            interactive=True,
                        )

                    duration_note = gr.Markdown(
                        value="✅ **Tối ưu**: Mốc **6s (97 frames @ 16 FPS)** an toàn và tối ưu cho RTX 5060 Ti.",
                        visible=True,
                    )

                # Card: Thông số nâng cao
                with gr.Accordion("🔧 Thông số nâng cao (Seed, Guidance, Frames)", open=False):
                    with gr.Row():
                        guidance_scale = gr.Slider(
                            label="Guidance Scale (CFG)",
                            minimum=1.0,
                            maximum=12.0,
                            step=0.5,
                            value=initial_values["guidance_scale"],
                        )
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=1000000,
                            step=1,
                            value=initial_values["seed"],
                        )
                    with gr.Row():
                        randomize_seed = gr.Checkbox(label="🎲 Randomize seed", value=False)
                        num_frames = gr.Number(
                            label="Number of Frames (4k + 1)",
                            value=initial_values["num_frames"],
                            interactive=True,
                        )
                    with gr.Row():
                        height = gr.Number(label="Render Height", value=initial_values["height"], interactive=True)
                        width = gr.Number(label="Render Width", value=initial_values["width"], interactive=True)

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
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                    <h3 style="margin: 0; color: #f0f6fc; font-size: 18px; font-weight: 700;">🎬 Danh Sách Video & Hàng Đợi Tiến Trình</h3>
                    <span style="font-size: 12px; color: #8b9bb4;">Tự động cập nhật từng dòng</span>
                </div>
                """)

                overall_status = gr.HTML(
                    value="<div class='banner-info'>⚡ Sẵn sàng nhận yêu cầu tạo video. Nhấn <strong>BẮT ĐẦU TẠO VIDEO</strong> để khởi chạy.</div>",
                    visible=True,
                )

                # Khởi tạo NUM_CARDS slots video cards dạng dòng
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
                            c_hdr = gr.Markdown(f"### 🎬 Video #{i+1}")
                            c_sta = gr.HTML("<span class='status-badge badge-idle'>Sẵn sàng</span>")
                        c_prg = gr.HTML(make_progress_bar_html(0, "Chờ bắt đầu...", "idle"))
                        c_vid = gr.Video(label=f"Video #{i+1}", height=380, interactive=False)
                        c_met = gr.Markdown(value="", visible=False)

                    card_groups.append(c_grp)
                    card_headers.append(c_hdr)
                    card_statuses.append(c_sta)
                    card_progresses.append(c_prg)
                    card_videos.append(c_vid)
                    card_metas.append(c_met)

                # Video Extension Panel
                with gr.Accordion("➕ Nối tiếp thêm +5s vào video đã tạo (Video Extension)", open=False):
                    extend_target = gr.Dropdown(
                        choices=[f"Video #{i+1}" for i in range(NUM_CARDS)],
                        value="Video #1",
                        label="🎯 Chọn video muốn nối tiếp",
                    )
                    extend_prompt = gr.Textbox(
                        label="Prompt phân cảnh nối tiếp (+5s)",
                        placeholder="Mô tả hành động tiếp theo (để trống sẽ tự động tiếp diễn theo ngữ cảnh)...",
                        lines=2,
                    )
                    extend_button = gr.Button("🎬 Nối tiếp phân cảnh (+5s)", variant="secondary")
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
            if "Tạo đơn" in mode:
                placeholder = "Nhập mô tả video bạn muốn tạo (Ví dụ: A majestic eagle soaring over snowy mountains at sunset, cinematic 4k)..."
                hint = "💡 **Chế độ Tạo đơn**: Cả khung văn bản trên sẽ được dùng để tạo ra **1 video duy nhất**."
            else:
                placeholder = (
                    "Nhập mỗi dòng là 1 prompt riêng biệt để tạo nhiều video liên tiếp:\n"
                    "Dòng 1: Chú mèo đội nón lá uống cà phê bên bờ hồ Hoàn Kiếm\n"
                    "Dòng 2: Phi thuyền không gian bay qua các hành tinh lấp lánh ánh sao\n"
                    "Dòng 3: Khung cảnh hoàng hôn rực rỡ trên bãi biển nhiệt đới..."
                )
                hint = "💡 **Chế độ Tạo nhiều video (Hàng đợi)**: Mỗi dòng sẽ được tạo thành **1 video riêng biệt** tuần tự trên GPU."
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

        def on_ratio_or_res_change(ratio: str, res: str):
            cfg = RESOLUTION_CONFIGS.get((ratio, res), (832, 448, 832, 448))
            native_w, native_h, target_w, target_h = cfg
            if (target_w, target_h) == (native_w, native_h):
                msg = f"📐 Độ phân giải xuất: **{target_w} x {target_h}** (Render gốc siêu tốc Wan2.1)"
            else:
                msg = f"📐 Độ phân giải xuất: **{target_w} x {target_h}** (Render gốc {native_w}x{native_h} -> Siêu phân giải Lanczos4 + Unsharp)"
            return gr.update(value=target_h), gr.update(value=target_w), gr.update(value=msg)

        aspect_ratio.change(
            fn=on_ratio_or_res_change,
            inputs=[aspect_ratio, resolution],
            outputs=[height, width, resolution_info],
        )

        resolution.change(
            fn=on_ratio_or_res_change,
            inputs=[aspect_ratio, resolution],
            outputs=[height, width, resolution_info],
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

            init_banner = f"<div class='banner-info'>📋 Hàng đợi: <strong>{num_runs} video</strong> cần tạo. Bắt đầu xử lý tuần tự...</div>"
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
                banner_now = f"<div class='banner-info'>⚡ Đang xử lý <strong>Video #{idx+1}/{num_runs}</strong>: <em>{cur_prompt[:60]}...</em></div>"
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

            model_path = MODEL_PATH_MAPPING.get(model_sel, "FastVideo/FastWan2.1-T2V-1.3B-Diffusers")
            generator = generators[model_path]
            params = deepcopy(default_params[model_path])

            res_cfg = RESOLUTION_CONFIGS.get((ratio, res), (832, 448, 832, 448))
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
