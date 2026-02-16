# GPT-SoVITS TTS Server - Linux 설치 가이드

## 사전 요구사항

- Ubuntu 20.04+ / Debian 11+ / CentOS 8+
- Python 3.10 또는 3.11
- NVIDIA GPU + CUDA 11.8 (권장)
- 디스크 공간: 약 15GB

---

## 1단계: 파일 준비 (Windows에서)

### 방법 A: 자동 패키징
```
Windows에서 pack-server.bat 실행
→ deploy_package 폴더 생성됨
```

### 방법 B: 수동 복사
필요한 파일/폴더:
```
├── tts_api_server.py      # API 서버
├── run-server.sh          # 실행 스크립트
├── install-linux.sh       # 설치 스크립트
├── requirements.txt       # Python 의존성
├── GPT_SoVITS/            # 코어 (전체)
├── tools/                 # 도구
└── presets/               # 프리셋 (선택)
```

---

## 2단계: 서버로 파일 전송

```bash
# SCP로 복사
scp -r deploy_package/ user@SERVER_IP:/home/user/gpt-sovits/

# 또는 rsync
rsync -avz deploy_package/ user@SERVER_IP:/home/user/gpt-sovits/
```

---

## 3단계: 서버에서 설치

### 3-1. SSH 접속
```bash
ssh user@SERVER_IP
cd /home/user/gpt-sovits
```

### 3-2. 실행 권한 부여
```bash
chmod +x install-linux.sh run-server.sh
```

### 3-3. 설치 스크립트 실행
```bash
./install-linux.sh
```

설치 내용:
- Python 가상환경 (venv)
- PyTorch + CUDA 11.8
- requirements.txt 의존성
- FastAPI, Uvicorn, Websockets

### 3-4. 수동 설치 (install-linux.sh 안될 경우)
```bash
# 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# PyTorch 설치 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 의존성 설치
pip install -r requirements.txt
pip install fastapi uvicorn websockets
```

---

## 4단계: SSL 인증서 설정 (WSS 사용 시)

### 인증서 경로 설정
`tts_api_server.py` 파일 수정:

```python
# 1488-1489번 줄 근처
default_ssl_key = "/home/user/ssl/key.pem"      # 수정
default_ssl_cert = "/home/user/ssl/cert.pem"    # 수정
```

### 인증서 복사
```bash
# STT 서버와 같은 인증서 사용
cp /path/to/key.pem /home/user/ssl/
cp /path/to/cert.pem /home/user/ssl/
```

---

## 5단계: 서버 실행

### 대화형 실행
```bash
./run-server.sh
```

모드 선택:
```
  [1] WS only (HTTP) - default
  [2] WSS only (HTTPS)
  [3] WS + WSS both

  Choice (1-3, default=1): 
```

### 직접 실행
```bash
source venv/bin/activate

# WS만
python tts_api_server.py --mode ws

# WSS만
python tts_api_server.py --mode wss

# 둘 다
python tts_api_server.py --mode both
```

---

## 6단계: 백그라운드 실행

### 방법 1: nohup
```bash
nohup ./run-server.sh > tts_server.log 2>&1 &

# 로그 확인
tail -f tts_server.log

# 종료
pkill -f tts_api_server.py
```

### 방법 2: screen
```bash
screen -S tts
./run-server.sh

# 분리: Ctrl+A, D
# 재연결: screen -r tts
# 종료: screen 내에서 Ctrl+C
```

### 방법 3: systemd 자동 설정 (권장)

자동 설정 스크립트 실행:
```bash
chmod +x setup_service.sh
sudo ./setup_service.sh
```

실행하면 모드 선택(WS/WSS/both) 후 자동으로:
- 서비스 파일 생성 (`/etc/systemd/system/gpt-sovits-tts.service`)
- systemd 등록
- 부팅 시 자동 시작 설정
- 서비스 시작
- **서버 다운 시 10초 후 자동 재시작**

### 방법 4: systemd 수동 설정

서비스 파일 복사:
```bash
sudo cp gpt-sovits-tts.service /etc/systemd/system/

# 경로 수정 (필요시)
sudo nano /etc/systemd/system/gpt-sovits-tts.service
```

서비스 등록 및 시작:
```bash
sudo systemctl daemon-reload
sudo systemctl enable gpt-sovits-tts
sudo systemctl start gpt-sovits-tts
```

유용한 명령어:
```bash
# 상태 확인
sudo systemctl status gpt-sovits-tts

# 로그 보기
sudo journalctl -u gpt-sovits-tts -f

# 재시작
sudo systemctl restart gpt-sovits-tts

# 중지
sudo systemctl stop gpt-sovits-tts

# 비활성화 (부팅 시 시작 안함)
sudo systemctl disable gpt-sovits-tts
```

---

## 7단계: 방화벽 설정

```bash
# UFW (Ubuntu)
sudo ufw allow 9874/tcp   # WS
sudo ufw allow 9875/tcp   # WSS

# firewalld (CentOS)
sudo firewall-cmd --permanent --add-port=9874/tcp
sudo firewall-cmd --permanent --add-port=9875/tcp
sudo firewall-cmd --reload
```

---

## 8단계: 테스트

### 헬스체크
```bash
curl http://localhost:9874/api/health
```

### API 문서
```
http://SERVER_IP:9874/docs
```

### 클라이언트 연결
```python
import websockets
import asyncio

async def test():
    uri = "ws://SERVER_IP:9874/ws/tts"  # 또는 wss://...9875
    async with websockets.connect(uri) as ws:
        await ws.send('{"text": "안녕하세요", "preset_id": "sample"}')
        # ... 응답 처리

asyncio.run(test())
```

---

## 포트 정리

| 모드 | 프로토콜 | 포트 | 용도 |
|------|----------|------|------|
| ws | HTTP | 9874 | 일반 연결 |
| wss | HTTPS | 9875 | 보안 연결 (브라우저 HTTPS용) |

---

## 문제 해결

### CUDA 오류
```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### 포트 사용 중
```bash
# 포트 확인
sudo lsof -i :9874

# 프로세스 종료
sudo kill -9 PID
```

### 권한 오류
```bash
# 폴더 권한
chmod -R 755 /home/user/gpt-sovits

# 실행 권한
chmod +x *.sh
```

### 메모리 부족
```bash
# GPU 메모리 확인
nvidia-smi

# 다른 프로세스 종료 또는 batch_size 줄이기
```
