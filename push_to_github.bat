@echo off
title Push FastVideo AI Studio Pro to GitHub (84-coder)
cd /d "%~dp0"

echo =====================================================================
echo   Pushing FastVideo AI Studio Pro to GitHub
echo   Target: https://github.com/84-coder/FastVideo-AI-Studio
echo =====================================================================
echo.
echo [1/4] Kiem tra Remote Git...

git remote get-url upstream >nul 2>&1
if errorlevel 1 (
    echo Luu remote goc hao-ai-lab thanh 'upstream'...
    git remote rename origin upstream >nul 2>&1
)

git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo Cai dat origin moi: https://github.com/84-coder/FastVideo-AI-Studio.git
    git remote add origin https://github.com/84-coder/FastVideo-AI-Studio.git
) else (
    echo Cap nhat origin sang: https://github.com/84-coder/FastVideo-AI-Studio.git
    git remote set-url origin https://github.com/84-coder/FastVideo-AI-Studio.git
)

echo.
echo [2/4] Kiem tra nhanh git branch...
git branch -M main

echo.
echo [3/4] Dong goi va tao commit san pham...
git add -A
git commit -m "feat: release FastVideo AI Studio Pro with batch queue, multi-scene chaining, and super-resolution"

echo.
echo [4/4] Dang day ma nguon len GitHub (origin main)...
echo Luu y: Neu chua tao repo tren GitHub, hay vao https://github.com/new de tao repo rong ten 'FastVideo-AI-Studio'.
echo.
git push -u origin main

if errorlevel 1 (
    echo.
    echo =====================================================================
    echo [!] Day len GitHub chua hoan tat.
    echo Nguyen nhan pho bien:
    echo 1. Ban chua tao repo 'FastVideo-AI-Studio' tren https://github.com/84-coder
    echo    -> Hay vao https://github.com/new va tao repo co ten 'FastVideo-AI-Studio'
    echo 2. Ban chua dang nhap Git tren may tinh (Personal Access Token / SSH Key)
    echo    -> Khi trinh duyet hien ra hoi quyen dang nhap GitHub, vui long chon Authorize.
    echo =====================================================================
) else (
    echo.
    echo =====================================================================
    echo [OK] DA DAY THANH CONG LEN:
    echo      https://github.com/84-coder/FastVideo-AI-Studio
    echo =====================================================================
)

pause
