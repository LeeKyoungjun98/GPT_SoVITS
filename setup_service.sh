#!/bin/bash
# ========================================
# GPT-SoVITS TTS systemd 서비스 자동 설정
# ========================================

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 현재 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="gpt-sovits-tts"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo "========================================"
echo "  GPT-SoVITS TTS systemd 서비스 설정"
echo "========================================"
echo

# root 권한 확인
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}sudo 권한이 필요합니다.${NC}"
    echo "다시 실행: sudo ./setup_service.sh"
    exit 1
fi

# 실제 사용자 확인 (sudo 실행 시 원래 사용자)
if [ -n "$SUDO_USER" ]; then
    ACTUAL_USER=$SUDO_USER
else
    ACTUAL_USER=$(whoami)
fi

# 가상환경 확인
if [ ! -f "$SCRIPT_DIR/venv/bin/python" ]; then
    echo -e "${RED}오류: 가상환경이 없습니다.${NC}"
    echo "먼저 install-linux.sh를 실행하세요."
    exit 1
fi

# 모드 선택
echo "서버 실행 모드 선택:"
echo "  [1] WS만 (HTTP) - 기본"
echo "  [2] WSS만 (HTTPS)"
echo "  [3] WS + WSS 둘 다"
echo
read -p "선택 (1-3, 기본=1): " MODE_CHOICE

case "$MODE_CHOICE" in
    2) SERVER_MODE="wss" ;;
    3) SERVER_MODE="both" ;;
    *) SERVER_MODE="ws" ;;
esac

echo
echo -e "${YELLOW}[1/4] 서비스 파일 생성 중...${NC}"

# 서비스 파일 생성
cat > "$SERVICE_FILE" << EOF
[Unit]
Description=GPT-SoVITS TTS Server
After=network.target

[Service]
User=$ACTUAL_USER
Group=$ACTUAL_USER
WorkingDirectory=$SCRIPT_DIR
Environment="PATH=$SCRIPT_DIR/venv/bin"
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="PYTHONUNBUFFERED=1"

ExecStart=$SCRIPT_DIR/venv/bin/python tts_api_server.py --mode $SERVER_MODE

Restart=always
RestartSec=10

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}서비스 파일 생성됨: $SERVICE_FILE${NC}"

echo
echo -e "${YELLOW}[2/4] systemd 데몬 리로드...${NC}"
systemctl daemon-reload
echo -e "${GREEN}완료${NC}"

echo
echo -e "${YELLOW}[3/4] 서비스 활성화 (부팅 시 자동 시작)...${NC}"
systemctl enable $SERVICE_NAME
echo -e "${GREEN}완료${NC}"

echo
echo -e "${YELLOW}[4/4] 서비스 시작...${NC}"
systemctl start $SERVICE_NAME
sleep 3

# 상태 확인
if systemctl is-active --quiet $SERVICE_NAME; then
    echo -e "${GREEN}✓ 서비스가 실행 중입니다!${NC}"
else
    echo -e "${RED}✗ 서비스 시작 실패${NC}"
    echo "로그 확인: sudo journalctl -u $SERVICE_NAME -n 50"
    exit 1
fi

echo
echo "========================================"
echo -e "${GREEN}  설정 완료!${NC}"
echo "========================================"
echo
echo "서비스 모드: $SERVER_MODE"
echo
echo "유용한 명령어:"
echo "  상태 확인:    sudo systemctl status $SERVICE_NAME"
echo "  로그 확인:    sudo journalctl -u $SERVICE_NAME -f"
echo "  재시작:       sudo systemctl restart $SERVICE_NAME"
echo "  중지:         sudo systemctl stop $SERVICE_NAME"
echo "  비활성화:     sudo systemctl disable $SERVICE_NAME"
echo
