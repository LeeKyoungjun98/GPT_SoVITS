#!/bin/bash
# ========================================
# GPT-SoVITS TTS 서버 자동 재시작 스크립트
# 서버가 죽으면 자동으로 다시 시작합니다
# ========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 가상환경 활성화
source venv/bin/activate

# 설정
MAX_RESTARTS=100          # 최대 재시작 횟수
RESTART_DELAY=5           # 재시작 대기 시간 (초)
SERVER_ARGS="--mode wss --version v2Pro"  # 서버 실행 옵션 (필요에 따라 수정)

restart_count=0

echo "========================================"
echo "  GPT-SoVITS TTS 자동 재시작 모드"
echo "========================================"
echo "  옵션: $SERVER_ARGS"
echo "  최대 재시작: ${MAX_RESTARTS}회"
echo "  재시작 대기: ${RESTART_DELAY}초"
echo "========================================"
echo

while [ $restart_count -lt $MAX_RESTARTS ]; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 서버 시작 (재시작 횟수: $restart_count)"
    
    python tts_api_server.py $SERVER_ARGS
    
    exit_code=$?
    
    # Ctrl+C로 종료한 경우 (정상 종료)
    if [ $exit_code -eq 130 ] || [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 서버 정상 종료 (exit code: $exit_code)"
        break
    fi
    
    restart_count=$((restart_count + 1))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ 서버 비정상 종료! (exit code: $exit_code)"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${RESTART_DELAY}초 후 재시작합니다... ($restart_count/$MAX_RESTARTS)"
    sleep $RESTART_DELAY
done

if [ $restart_count -ge $MAX_RESTARTS ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ 최대 재시작 횟수 초과! 서버를 확인해주세요."
fi
