@echo off
title FastVideo AI Studio Pro
cd /d "%~dp0"

echo ========================================================
echo   Starting FastVideo AI Studio Pro...
echo ========================================================

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set MASTER_ADDR=127.0.0.1
set FASTVIDEO_LOOPBACK_IP=127.0.0.1
set FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN

call .venv\Scripts\activate.bat
python run_studio.py --port 7860 --host 127.0.0.1

pause
