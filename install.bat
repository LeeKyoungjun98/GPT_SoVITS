@echo off
echo ============================================
echo   GPT-SoVITS TTS Server - Install
echo ============================================
echo.

if not exist venv (
    echo [1/3] Creating virtual environment...
    python -m venv venv
)

echo [2/3] Installing PyTorch (CUDA 11.8)...
.\venv\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo [3/3] Installing dependencies...
.\venv\Scripts\pip.exe install -r requirements.txt
.\venv\Scripts\pip.exe install fastapi uvicorn websockets

echo.
echo ============================================
echo   Install complete!
echo   Run: run-api-server.bat
echo ============================================
pause
