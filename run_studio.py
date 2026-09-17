"""
FastVideo AI Studio Pro - Production Launcher
=============================================
Usage:
    python run_studio.py [--port 7860] [--host 127.0.0.1]
"""

import argparse
import os
import sys

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure IPv4 loopback on Windows to prevent error 10049 and configure CUDA allocator
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("FASTVIDEO_LOOPBACK_IP", "127.0.0.1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gradio as gr
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

from fastvideo.entrypoints.video_generator import VideoGenerator
from fastvideo.api.sampling_param import SamplingParam
from studio.app import create_studio_interface, setup_model_environment


def main():
    parser = argparse.ArgumentParser(description="FastVideo AI Studio Pro Launcher")
    parser.add_argument(
        "--t2v_model_paths",
        type=str,
        default="FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        help="Comma-separated list of pretrained model paths or HF IDs",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument(
        "--text_encoder_cpu_offload",
        action="store_true",
        default=True,
        help="Offload text encoder to CPU (recommended for <= 24GB VRAM)",
    )
    parser.add_argument(
        "--no_text_encoder_cpu_offload",
        dest="text_encoder_cpu_offload",
        action="store_false",
    )
    parser.add_argument(
        "--dit_layerwise_offload",
        action="store_true",
        default=False,
        help="Enable DiT layerwise offload (for minimal VRAM systems)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  🎬 Khởi động FastVideo AI Studio Pro...")
    print(f"  🌐 URL: http://{args.host}:{args.port}")
    print("=" * 60)

    generators = {}
    default_params = {}
    model_paths = [p.strip() for p in args.t2v_model_paths.split(",") if p.strip()]

    import torch
    total_vram_gb = 0.0
    if torch.cuda.is_available():
        try:
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            pass

    for model_path in model_paths:
        print(f"📦 Đang tải mô hình: {model_path}")
        setup_model_environment(model_path)
        is_5b = ("5B" in model_path or "5b" in model_path)
        need_layerwise = args.dit_layerwise_offload or is_5b or (total_vram_gb > 0 and total_vram_gb <= 10)
        generators[model_path] = VideoGenerator.from_pretrained(
            model_path,
            num_gpus=1,
            text_encoder_cpu_offload=args.text_encoder_cpu_offload,
            dit_layerwise_offload=need_layerwise,
            dit_cpu_offload=False,
            vae_cpu_offload=True,
        )
        default_params[model_path] = SamplingParam.from_pretrained(model_path)

    demo = create_studio_interface(default_params, generators)

    app = FastAPI(title="FastVideo AI Studio Pro")

    @app.get("/logo.png")
    def get_logo():
        logo_path = "assets/full.svg"
        if os.path.exists(logo_path):
            return FileResponse(
                logo_path,
                media_type="image/svg+xml",
                headers={"Cache-Control": "public, max-age=3600", "Access-Control-Allow-Origin": "*"},
            )
        raise HTTPException(status_code=404, detail="Logo not found")

    @app.get("/favicon.ico")
    def get_favicon():
        favicon_path = "assets/icon-simple.svg"
        if os.path.exists(favicon_path):
            return FileResponse(
                favicon_path,
                media_type="image/svg+xml",
                headers={"Cache-Control": "public, max-age=3600", "Access-Control-Allow-Origin": "*"},
            )
        raise HTTPException(status_code=404, detail="Favicon not found")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        base_url = str(request.base_url).rstrip("/")
        return f"""
        <!DOCTYPE html>
        <html lang="vi">
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <title>FastVideo AI Studio Pro</title>
            <meta name="description" content="Hệ thống tạo video AI chuyên nghiệp cao cấp">
            <link rel="icon" type="image/svg+xml" href="/favicon.ico">
            <style>
                body, html {{
                    margin: 0;
                    padding: 0;
                    height: 100%;
                    overflow: hidden;
                    background: #0f141c;
                }}
                iframe {{
                    width: 100%;
                    height: 100vh;
                    border: none;
                }}
            </style>
        </head>
        <body>
            <iframe src="/gradio" width="100%" height="100%"></iframe>
        </body>
        </html>
        """

    os.makedirs("outputs", exist_ok=True)
    allowed = [os.path.abspath("outputs")]
    if os.path.exists("assets"):
        allowed.append(os.path.abspath("assets"))

    app = gr.mount_gradio_app(app, demo, path="/gradio", allowed_paths=allowed)

    print(f"\n🚀 Studio đã sẵn sàng! Mở trình duyệt tại: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
