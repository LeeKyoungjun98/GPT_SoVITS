#!/bin/bash
echo "============================================"
echo "  GPT-SoVITS TTS Server - Linux 설치"
echo "============================================"
echo

# 현재 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Python 버전 확인
PYTHON_CMD=""
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python 3.10+ 가 필요합니다."
    exit 1
fi

echo "Python: $PYTHON_CMD ($($PYTHON_CMD --version))"
echo

# 가상환경 생성
if [ ! -d "venv" ]; then
    echo "[1/4] 가상환경 생성 중..."
    $PYTHON_CMD -m venv venv
else
    echo "[1/4] 가상환경 이미 존재함"
fi

# 가상환경 활성화
source venv/bin/activate

# PyTorch 설치
echo "[2/4] PyTorch 설치 중 (CUDA 11.8)..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 의존성 설치
echo "[3/4] 의존성 설치 중..."
pip install -r requirements.txt

# API 서버용 추가 패키지
echo "[4/4] FastAPI 설치 중..."
pip install fastapi uvicorn websockets

echo
echo "============================================"
echo "  설치 완료!"
echo "============================================"
echo
echo "서버 실행 방법:"
echo "  source venv/bin/activate"
echo "  python tts_api_server.py"
echo
echo "또는:"
echo "  ./run-server.sh"
echo
