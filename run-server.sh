#!/bin/bash
echo "============================================"
echo "  GPT-SoVITS TTS API Server"
echo "============================================"
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 가상환경 활성화
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: 가상환경이 없습니다. install-linux.sh를 먼저 실행하세요."
    exit 1
fi

# 명령줄 인자가 있으면 바로 실행
if [ $# -gt 0 ]; then
    echo "Press Ctrl+C to stop"
    echo
    python tts_api_server.py "$@"
    exit 0
fi

# 모드 선택
echo "  실행 모드 선택:"
echo
echo "    [1] WS만 (HTTP) - 기본"
echo "    [2] WSS만 (HTTPS) - 브라우저 HTTPS용"
echo "    [3] WS + WSS 둘 다"
echo
read -p "  선택 (1-3, 기본=1): " MODE_CHOICE

case "$MODE_CHOICE" in
    2) MODE="wss" ;;
    3) MODE="both" ;;
    *) MODE="ws" ;;
esac

echo
echo "============================================"
echo "Starting API server (mode: $MODE)..."
echo "Press Ctrl+C to stop"
echo

python tts_api_server.py --mode $MODE
