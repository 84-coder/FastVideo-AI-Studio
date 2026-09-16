<div align="center">
  <img src="assets/logos/logo.svg" width="30%" alt="FastVideo Logo"/>
  <h1>🎬 FastVideo AI Studio Pro</h1>
  <p><b>Hệ thống sản xuất video AI thương mại chuyên nghiệp thế hệ mới</b></p>
  <p><i>High-Throughput Cinematic AI Video Generation Studio powered by Wan2.1 Diffusion Transformers, Triton Sparse Attention & Lanczos4 Super-Resolution.</i></p>
</div>

---

## 🌟 Giới thiệu (Overview)

**FastVideo AI Studio Pro** là bộ giải pháp toàn diện biến công nghệ sinh video của Wan2.1 và FastVideo thành một studio sản xuất video điện ảnh thực thụ. 

Không chỉ dừng lại ở các demo kỹ thuật ngắn, **Studio Pro** được thiết kế phục vụ quy trình sản xuất nội dung chuyên nghiệp (YouTube, Shorts, TikTok, Quảng cáo TVC, Phim hoạt hình):
- 🚀 **Thời lượng dài (15s – 60s)** nhờ kiến trúc **Nối tiếp đa cảnh (Multi-Scene Chaining)** chống tràn VRAM.
- 📐 **Đa tỷ lệ khung hình chuẩn**: 16:9 (YouTube/Điện ảnh), 9:16 (Shorts/TikTok/Reels), 1:1 (Instagram/Vuông).
- ✨ **Siêu phân giải Lanczos4 + Unsharp Mask**: Xuất video sắc nét SD (480p), HD (720p), Full HD (1080p).
- 📋 **Hàng đợi tạo hàng loạt (Batch Queue)**: Nhập danh sách prompt theo từng dòng để hệ thống tự động render tuần tự trên GPU kèm thanh tiến trình trực tiếp.
- ➕ **Nối tiếp phân cảnh (+5s)**: Cho phép nối tiếp một video đã có để tạo chuỗi hành động liền mạch.
- ⚡ **Tối ưu Windows & CUDA 12**: Khắc phục triệt để lỗi IPC socket, tự động cấu hình Triton Video Sparse Attention trên các dòng GPU NVIDIA RTX 40/50 Series.

---

## 🖥️ Giao diện Studio (Studio Architecture)

Hệ thống được thiết kế theo chuẩn Dark Obsidian Theme hiện đại:
```
+-----------------------------------------------------------------------------+
|                      🎬 FastVideo AI Studio Pro                             |
+------------------------------------+----------------------------------------+
| 📝 CỘT CẤU HÌNH & NHẬP LIỆU        | 🎬 DANH SÁCH VIDEO & TIẾN TRÌNH        |
|                                    |                                        |
| 🤖 Model: FastWan2.1-T2V-1.3B      | ⚡ Sẵn sàng nhận lệnh / Hàng đợi       |
| 💡 Presets: [Chọn Prompt Mẫu...]   |                                        |
| 🎯 Mode: [Tạo Đơn] / [Tạo Hàng Đợi]| [🎬 Video #1: Hoàng hôn tuyết...]      |
| 📝 Prompt Textarea (Multi-line)    | [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 100%] Hoàn thành |
| 🚫 Negative Prompt                 | [▶️ Trình phát Video Player #1]        |
|                                    |                                        |
| ⚙️ Video Options:                  | [🎬 Video #2: Chiến binh Cyberpunk...] |
| - Tỷ lệ: 16:9 / 9:16 / 1:1         | [▓▓▓▓▓▓▓▓▓▓░░░░░░░░░ 50%] GPU Render   |
| - Phân giải: SD / HD / Full HD     | [▶️ Trình phát Video Player #2]        |
| - Chế độ: Đơn cảnh / Đa cảnh       |                                        |
| - Thời lượng: 3s, 5s, 6s... đến 60s| ➕ Nối tiếp phân cảnh (+5s Extension)  |
| 🔧 CFG Guidance, Seed, Frames      | - Chọn video nguồn -> Nối tiếp +5s     |
| [🚀 BẮT ĐẦU TẠO VIDEO (GENERATE)]  |                                        |
+------------------------------------+----------------------------------------+
```

---

## ⚡ Khởi động nhanh (Quick Start)

### 1. Khởi chạy bằng 1 click trên Windows:
Chỉ cần nhấp đúp vào file:
```bat
run_studio.bat
```
Hoặc chạy lệnh từ terminal:
```bash
python run_studio.py --port 7860 --host 127.0.0.1
```
Trình duyệt sẽ tự động phục vụ tại địa chỉ: **http://127.0.0.1:7860**

---

## 📂 Cấu trúc mã nguồn Product (Codebase Structure)

```
FastVideo-AI-Studio/
├── studio/                           # Module lõi của AI Studio Pro
│   ├── __init__.py                   # Khởi tạo package studio
│   ├── config.py                     # Cấu hình phân giải, tỷ lệ, thời lượng, 4k+1 mapping
│   ├── postprocess.py                # Siêu phân giải Lanczos4, ghép video, nối cảnh
│   ├── ui_helpers.py                 # Bộ đo thời gian, thanh tiến trình, stylesheet
│   ├── app.py                        # Giao diện Gradio 2 cột & bộ xử lý hàng đợi
│   └── prompts.txt                   # Thư viện prompt mẫu chuẩn điện ảnh
│
├── run_studio.py                     # Entrypoint chính khởi chạy Studio
├── run_studio.bat                    # Script khởi chạy 1-click trên Windows
├── push_to_github.bat                # Script đẩy mã nguồn lên GitHub cá nhân
│
├── fastvideo/                        # Lõi tăng tốc FastVideo & Wan2.1 Engine
│   ├── api/                          # Sampling params & configs
│   ├── entrypoints/                  # VideoGenerator runtime
│   ├── models/                       # Kiến trúc Wan2.1 DiT, Text Encoders, VAE
│   ├── pipelines/                    # Quy trình xử lý đa giai đoạn
│   └── attention/                    # Video Sparse Attention (Triton)
│
└── outputs/                          # Thư mục xuất video thành phẩm (.mp4)
```

---

## 💡 Hướng dẫn Đẩy lên GitHub Cá Nhân (Push to GitHub)

Để đẩy toàn bộ bộ mã nguồn Studio này lên tài khoản GitHub của bạn (`https://github.com/84-coder/FastVideo-AI-Studio`):

1. Truy cập [github.com/new](https://github.com/new) và tạo một repository mới có tên:
   **`FastVideo-AI-Studio`** (để chế độ Public hoặc Private tùy ý, không cần tích chọn README hay .gitignore vì repo cục bộ đã có sẵn).
2. Chạy script tự động:
   ```bat
   push_to_github.bat
   ```
   Script sẽ tự động:
   - Đổi Remote sang: `https://github.com/84-coder/FastVideo-AI-Studio.git`
   - Tạo commit chuẩn hóa `FastVideo AI Studio Pro`
   - Đẩy toàn bộ nhánh `main` lên GitHub của bạn.

---

## 🏆 Tính năng Nổi bật (Core Features)

### 1. Chuẩn công nghiệp: Video Dài 15s – 60s Không Tràn VRAM
Thay vì cố ép GPU render hàng trăm frame cùng một lúc dẫn đến lỗi Out Of Memory (OOM) hoặc chất lượng bị biến dạng, **FastVideo AI Studio Pro** áp dụng chuẩn Studio:
- Chia nhỏ kịch bản thành từng phân cảnh 5s (81 frames).
- Tự động thay đổi góc máy điện ảnh (*cinematic camera moves: wide establishing, tracking shot, close-up, dramatic low angle*).
- Ghép nối liền mạch các phân cảnh và nâng cấp chất lượng bằng thuật toán nội suy Lanczos4.

### 2. Tối ưu hóa GPU & Chế độ Tiết kiệm Bộ nhớ
- Hỗ trợ `--text_encoder_cpu_offload` (bật mặc định) cho phép các card đồ họa tầm trung (như RTX 4060, RTX 5060 Ti, RTX 3070/3080 8GB-16GB) hoạt động mượt mà.
- Hỗ trợ `--dit_layerwise_offload` cho các cấu hình RAM/VRAM giới hạn.

---

## 📜 Giấy phép & Lời cảm ơn (Credits & License)

- Engine gốc được phát triển bởi [Hao AI Lab (FastVideo)](https://github.com/hao-ai-lab/FastVideo).
- Kiến trúc mô hình: [Wan2.1](https://github.com/Wan-Video/Wan2.1) bởi Alibaba Wan-AI.
- Bản phân phối **FastVideo AI Studio Pro** được tùy biến và đóng gói bởi [84-coder](https://github.com/84-coder).
