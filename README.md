<div align="center">
  <img src="assets/logos/logo.svg" width="32%" alt="FastVideo AI Studio Pro Logo"/>
  <h1>🎬 FastVideo AI Studio Pro</h1>
  <p><b>Hệ Thống Sản Xuất Video AI Thương Mại Chuyên Nghiệp • Enterprise AI Video Studio</b></p>
  <p><i>High-Throughput Cinematic AI Video Generation powered by Wan2.1 Diffusion Transformers, Triton Video Sparse Attention & Lanczos4 Super-Resolution.</i></p>

  <p>
    <a href="https://github.com/84-coder/FastVideo-AI-Studio"><img src="https://img.shields.io/badge/GitHub-84--coder%2FFastVideo--AI--Studio-blue?style=for-the-badge&logo=github" alt="GitHub Repo"/></a>
    <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
    <img src="https://img.shields.io/badge/PyTorch-CUDA%2012%20%7C%20CUDA%2013-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
    <img src="https://img.shields.io/badge/GPU-RTX%2040%20%7C%20RTX%2050%20%7C%20H100-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="NVIDIA"/>
    <img src="https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge" alt="License"/>
  </p>
</div>

---

## 📑 Mục Lục (Table of Contents)

1. [🌟 Giới thiệu tổng quan (Overview)](#-giới-thiệu-tổng-quan-overview)
2. [✨ Tính năng đột phá (Key Highlights)](#-tính-năng-đột-phá-key-highlights)
3. [🖥️ Kiến trúc giao diện Studio (Studio Architecture)](#️-kiến-trúc-giao-diện-studio-studio-architecture)
4. [⚡ Cài đặt & Khởi động nhanh (Quick Start)](#-cài-đặt--khởi-động-nhanh-quick-start)
5. [🎬 Hướng dẫn sản xuất chuyên nghiệp (Production Guide)](#-hướng-dẫn-sản-xuất-chuyên-nghiệp-production-guide)
6. [📊 Đo lường hiệu năng thực tế (Hardware Benchmarks)](#-đo-lường-hiệu-năng-thực-tế-hardware-benchmarks)
7. [📂 Cấu trúc mã nguồn sản phẩm (Codebase Structure)](#-cấu-trúc-mã-nguồn-sản-phẩm-codebase-structure)
8. [🌐 Hướng dẫn đồng bộ GitHub (Push to GitHub)](#-hướng-dẫn-đồng-bộ-github-push-to-github)
9. [📜 Lời cảm ơn & Bản quyền (Credits & License)](#-lời-cảm-ơn--bản-quyền-credits--license)

---

## 🌟 Giới thiệu tổng quan (Overview)

**FastVideo AI Studio Pro** (`https://github.com/84-coder/FastVideo-AI-Studio`) là một bộ giải pháp toàn diện biến các mô hình sinh video tiên tiến (Wan2.1 DiT, FastWan) thành **một studio sản xuất nội dung AI thương mại hoàn chỉnh**, loại bỏ hoàn toàn các hạn chế của những bản demo kỹ thuật sơ khai.

Hệ thống được thiết kế đặc biệt cho các nhà sáng tạo nội dung YouTube, TikTok, Reels, đạo diễn TVC, và các studio làm phim hoạt hình/điện ảnh với khả năng:
- Tạo video dài từ **15s đến 60s** liền mạch mà không bị tràn bộ nhớ VRAM.
- Xuất đa tỷ lệ khung hình (**16:9, 9:16, 1:1**) và siêu phân giải lên tới **Full HD (1080p)**.
- Xử lý hàng đợi hàng loạt (Batch Queue) tự động render từng dòng prompt trên GPU.
- Tối ưu hóa sâu cho Windows 11 và dòng card đồ họa thế hệ mới NVIDIA RTX (Blackwell RTX 50 Series, Ada Lovelace RTX 40 Series).

---

## ✨ Tính năng đột phá (Key Highlights)

| Tính năng | Mô tả chi tiết |
|---|---|
| 🎬 **Nối Tiếp Đa Cảnh (Multi-Scene Studio)** | Tự động phân rã kịch bản thành các phân cảnh 5s, chuyển động góc máy điện ảnh linh hoạt (*establishing wide, tracking, close-up, low-angle*) và ghép lại thành video dài **15s – 60s** không lo tràn VRAM. |
| 📋 **Hàng Đợi Hàng Loạt (Batch Queue)** | Chế độ *"Mỗi dòng 1 Video"* cho phép nhập danh sách kịch bản; hệ thống sẽ render tuần tự, cập nhật tiến trình trực tiếp cho từng thẻ video. |
| 📐 **Đa Tỷ Lệ Chuẩn (Triple Aspect Ratios)** | Hỗ trợ **16:9 (Ngang - YouTube/TVC)**, **9:16 (Dọc - TikTok/Shorts/Reels)** và **1:1 (Vuông - Social Feed)** với kích thước latent căn chỉnh toán học chuẩn $4k+1$. |
| ✨ **Siêu Phân Giải (Super-Resolution)** | Thuật toán nội suy Lanczos4 kết hợp Unsharp Masking tái tạo tần số cao, nâng chất lượng xuất lên **HD (720p)** và **Full HD (1080p)** sắc nét. |
| ➕ **Nối Tiếp Phân Cảnh (+5s Extension)** | Cho phép chọn bất kỳ video nào trong danh sách đã tạo để nối dài thêm 5s hành động tiếp nối theo ngữ cảnh. |
| ⚡ **Triton Video Sparse Attention (VSA)** | Tăng tốc tính toán Self-Attention lên tới **>50x** so với Attention truyền thống, chạy 100% bản địa trên GPU không dùng fallback suy giảm chất lượng. |
| 🛡️ **Windows & CUDA 12/13 Native** | Khắc phục triệt để lỗi IPC socket Windows, tối ưu hóa giải phóng bộ nhớ UMT5 giúp giảm thời gian giải mã VAE từ 7 phút xuống **13 giây**. |

---

## 🖥️ Kiến trúc giao diện Studio (Studio Architecture)

Giao diện được thiết kế theo tiêu chuẩn **Dark Obsidian Theme** (`#0f141c` / `#161f2c`), bố cục 2 cột năng suất cao:

```
+-----------------------------------------------------------------------------------------+
|                              🎬 FastVideo AI Studio Pro                                 |
+-------------------------------------------+---------------------------------------------+
| 📝 CỘT TRÁI: CẤU HÌNH & NHẬP LIỆU         | 🎬 CỘT PHẢI: DANH SÁCH VIDEO & HÀNG ĐỢI     |
|                                           |                                             |
| 🤖 Model: FastWan2.1-T2V-1.3B             | ⚡ Banner trạng thái tổng thể hệ thống      |
| 💡 Presets: [Chọn Prompt Mẫu Điện Ảnh]    |                                             |
| 🎯 Chế độ: [Tạo Đơn] / [Tạo Hàng Đợi]     | [🎬 Video #1: Hoàng hôn tuyết rực rỡ...]    |
| 📝 Khung nhập Prompt kịch bản (8 dòng)    |   Status: ✅ Hoàn thành                     |
| 🚫 Negative Prompt (Tùy chọn lọc chi tiết)|   Progress: [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 100%]     |
|                                           |   [▶️ Trình phát Video Player #1]           |
| ⚙️ Cấu hình Video:                         |   Metadata: ⏱️ 60.5s | 📐 1920x1080 | 🌱 1024|
| - 📐 Tỷ lệ: 16:9 / 9:16 / 1:1             |                                             |
| - 📺 Phân giải: SD (480p) / HD / Full HD  | [🎬 Video #2: Cyberpunk Tokyo Mưa Đêm...]   |
| - 🎬 Chế độ: Đơn cảnh / Đa cảnh (15s-60s) |   Status: ⚡ Đang xử lý GPU (50%)...        |
| - ⏱️ Thời lượng: 3s, 5s, 6s... đến 60s    |   Progress: [▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░ 50%]      |
| 🔧 CFG Guidance, Seed, Frames, Kích thước |                                             |
|                                           | ➕ Nối tiếp phân cảnh (+5s Extension)       |
| [🚀 BẮT ĐẦU TẠO VIDEO (GENERATE)]         | - Chọn video nguồn -> Bấm nối tiếp +5s      |
+-------------------------------------------+---------------------------------------------+
```

---

## ⚡ Cài đặt & Khởi động nhanh (Quick Start)

### 1. Yêu cầu phần cứng & môi trường
- **Hệ điều hành**: Windows 10/11 (64-bit) hoặc Linux (Ubuntu 22.04+).
- **GPU**: NVIDIA GPU có $\ge 8\text{GB}$ VRAM (Tối ưu từ RTX 3060, RTX 4060, RTX 5060 Ti, RTX 4080/4090, A100/H100).
- **RAM**: Khuyến nghị $\ge 32\text{GB}$ RAM (tốt nhất 64GB).
- **Python**: Phiên bản 3.10, 3.11 hoặc 3.12.

### 2. Tải mã nguồn về máy
```bash
git clone https://github.com/84-coder/FastVideo-AI-Studio.git
cd FastVideo-AI-Studio
```

### 3. Cài đặt môi trường bằng `uv` (Khuyên dùng)
```bash
# Tạo môi trường ảo uv
uv venv --python 3.12 --seed
source .venv/bin/activate  # Trên Linux
# Hoặc trên Windows:
.venv\Scripts\activate.bat

# Cài đặt gói với hỗ trợ CUDA 12/13
UV_TORCH_BACKEND=cu126 uv pip install -e ".[dev]"
```

### 4. Khởi chạy Studio

#### 👉 Trên Windows (1-Click Launcher):
Chỉ cần nhấp đúp vào file:
```bat
run_studio.bat
```

#### 👉 Qua Command Line:
```bash
python run_studio.py --port 7860 --host 127.0.0.1
```
Mở trình duyệt truy cập: **`http://127.0.0.1:7860`**.

---

## 🎬 Hướng dẫn sản xuất chuyên nghiệp (Production Guide)

### 1. Chế độ Đơn Cảnh (Single-Shot 3s – 10s)
- Phù hợp để thử nghiệm các chuyển động micro-action, tạo stock footage hoặc phân cảnh cắt nhanh.
- Mốc đề xuất: **6s (97 frames @ 16 FPS)** — tối ưu dung lượng VRAM và giữ tính nhất quán vật lý tốt nhất.

### 2. Chế độ Đa Cảnh Studio (Multi-Scene 15s – 60s)
- Khi chọn chế độ **Nối tiếp đa cảnh**, bạn có thể nhập các phân đoạn kịch bản cách nhau bằng dấu ba gạch `---` hoặc xuống dòng kép `\n\n`:
  ```text
  Cảnh toàn: Một phi thuyền không gian bay qua các vành đai sao Thổ rực rỡ ánh sáng mặt trời
  ---
  Cảnh trung: Bên trong buồng lái, nữ phi hành gia đang điều chỉnh các nút bấm hologram phát sáng
  ---
  Cảnh cận: Đôi mắt của phi hành gia phản chiếu hình ảnh một tinh vân xanh biếc kỳ vĩ phía trước
  ```
- Studio sẽ render tuần tự từng cảnh 5s và tự động nối lại thành một video hoàn chỉnh với độ phân giải cao!

### 3. Hàng Đợi Nhiều Video (Batch Queue)
- Chọn chế độ **`📋 Tạo nhiều video (Hàng đợi)`**.
- Mỗi dòng nhập một prompt riêng biệt. Studio sẽ xử lý lần lượt từng video trên GPU; bạn có thể xem và tải ngay video vừa hoàn thành trong khi GPU đang tiếp tục render video tiếp theo.

---

## 📊 Đo lường hiệu năng thực tế (Hardware Benchmarks)

*Kiểm nghiệm thực tế trên hệ thống NVIDIA GeForce RTX 5060 Ti (16GB VRAM, Windows 11, PyTorch 2.12 + cu130):*

| Giai đoạn Pipeline | Thời gian xử lý | Mức tiêu thụ VRAM | Ghi chú kỹ thuật |
|---|---|---|---|
| **Text Encoding (UMT5-XXL)** | 11.8 s | ~11.5 GB | Tự động offload sang RAM sau khi xong |
| **DMD Denoising (3 bước)** | 32.3 s | ~4.2 GB | Tăng tốc qua Triton Video Sparse Attention (~10.7s/bước) |
| **VAE Decoding (61 frames)** | 13.3 s | ~12.4 GB | 100% GPU resident, không bị phân trang PCIe |
| **Siêu phân giải Lanczos4** | 3.2 s | System RAM | Nâng cấp sắc nét lên 720p / 1080p |
| **Tổng thời gian End-to-End** | **~60.5 giây** | **12.4 GB Peak** | **Chạy mượt mà, hoàn toàn không tràn VRAM** |

---

## 📂 Cấu trúc mã nguồn sản phẩm (Codebase Structure)

```
FastVideo-AI-Studio/
├── studio/                           # 📦 Module lõi Studio Pro độc lập
│   ├── __init__.py                   # Khởi tạo package studio
│   ├── config.py                     # Cấu hình phân giải, tỷ lệ, thời lượng, 4k+1 frame mapping
│   ├── postprocess.py                # Siêu phân giải Lanczos4 + Unsharp, ghép video, nối cảnh
│   ├── ui_helpers.py                 # Bộ đo thời gian, tạo HTML progress bar động, Dark Theme
│   ├── app.py                        # Giao diện Gradio 2 cột studio & bộ xử lý hàng đợi
│   └── prompts.txt                   # Thư viện prompt mẫu điện ảnh chất lượng cao
│
├── run_studio.py                     # 🚀 Entrypoint chính khởi chạy Studio (FastAPI + Uvicorn)
├── run_studio.bat                    # ⚡ Script 1-click khởi chạy trên Windows
├── push_to_github.bat                # 🌐 Script tự động đẩy repo lên GitHub 84-coder
├── README.md                         # 📖 Tài liệu sản phẩm hoàn chỉnh
│
├── fastvideo/                        # 🧠 Lõi tăng tốc Wan2.1 DiT & Triton Sparse Attention
│   ├── api/                          # Sampling params & configs
│   ├── entrypoints/                  # VideoGenerator runtime
│   ├── models/                       # Kiến trúc Wan2.1 DiT, Text Encoders, VAE
│   ├── pipelines/                    # Quy trình xử lý đa giai đoạn (Text, DMD, VAE)
│   └── attention/                    # Triton Video Sparse Attention
│
└── outputs/                          # 📁 Thư mục lưu trữ video thành phẩm (.mp4)
```

---

## 🌐 Hướng dẫn đồng bộ GitHub (Push to GitHub)

Repository này được liên kết trực tiếp với tài khoản **`84-coder`**:
👉 **`https://github.com/84-coder/FastVideo-AI-Studio`**

### Quy trình đẩy mã nguồn lên GitHub của bạn:
1. Vào **[github.com/new](https://github.com/new)** và tạo một repository rỗng với tên:
   **`FastVideo-AI-Studio`**
2. Nhấp đúp vào file:
   ```bat
   push_to_github.bat
   ```
Script sẽ tự động:
- Trỏ remote `origin` về `https://github.com/84-coder/FastVideo-AI-Studio.git`.
- Tự động đóng gói commit sạch sẽ.
- Đẩy toàn bộ nhánh `main` lên repository của bạn.

---

## 📜 Lời cảm ơn & Bản quyền (Credits & License)

- **Engine nền tảng**: Phát triển bởi nhóm nghiên cứu [Hao AI Lab (FastVideo)](https://github.com/hao-ai-lab/FastVideo).
- **Kiến trúc mô hình Diffusion**: [Wan2.1](https://github.com/Wan-Video/Wan2.1) bởi Alibaba Wan-AI.
- **Bản phân phối Studio Pro**: Thiết kế, hoàn thiện kiến trúc đa cảnh, giao diện hàng đợi và đóng gói bởi [84-coder](https://github.com/84-coder).
- Giấy phép mã nguồn: **Apache License 2.0**.
