# GPT-SoVITS TTS API 서버 배포 가이드

## 필요한 파일/폴더

### 1. 핵심 파일 (필수)
```
📁 서버 폴더/
├── tts_api_server.py          # API 서버 (핵심)
├── run-api-server.bat         # 서버 실행 스크립트
├── requirements.txt           # Python 의존성
└── config.py                  # 설정 파일
```

### 2. GPT-SoVITS 코어 (필수)
```
📁 GPT_SoVITS/                 # 전체 폴더 복사
   ├── TTS_infer_pack/
   ├── module/
   ├── text/
   ├── feature_extractor/
   └── pretrained_models/      # 사전학습 모델
       ├── gsv-v2final-pretrained/
       ├── v2Pro/              # v2Pro 사용시
       ├── chinese-roberta-wwm-ext-large/
       └── chinese-hubert-base/
```

### 3. 프리셋 (옵션)
```
📁 presets/
   ├── presets.json            # 프리셋 설정
   └── *.wav                   # 참조 오디오 파일들
```

### 4. 도구 폴더 (필수)
```
📁 tools/
   └── i18n/                   # 다국어 지원
       ├── i18n.py
       └── locale/
```

---

## 설치 방법 (Windows)

### 1단계: 파일 복사
서버 컴퓨터에 위 파일들을 복사합니다.

### 2단계: Python 환경 설정
```powershell
# Python 3.10 또는 3.11 권장
cd 설치경로

# 가상환경 생성
python -m venv venv

# 가상환경 활성화
.\venv\Scripts\activate

# PyTorch 설치 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 의존성 설치
pip install -r requirements.txt

# API 서버용 추가 패키지
pip install fastapi uvicorn websockets
```

### 3단계: 모델 다운로드
pretrained_models가 없으면 다운로드:
- https://huggingface.co/lj1995/GPT-SoVITS

### 4단계: 서버 실행
```powershell
run-api-server.bat
```

또는 직접 실행:
```powershell
.\venv\Scripts\python.exe tts_api_server.py
```

---

## 설치 방법 (Linux)

### 1단계: 파일 복사
```bash
# SCP로 복사 예시
scp -r deploy_package/ user@server:/home/user/gpt-sovits/
```

### 2단계: Python 환경 설정
```bash
# Python 3.10 또는 3.11 권장
cd /home/user/gpt-sovits

# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# PyTorch 설치 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 의존성 설치
pip install -r requirements.txt

# API 서버용 추가 패키지
pip install fastapi uvicorn websockets
```

### 3단계: 서버 실행
```bash
# 직접 실행
source venv/bin/activate
python tts_api_server.py

# 또는 백그라운드 실행
nohup python tts_api_server.py > tts_server.log 2>&1 &

# 또는 systemd 서비스로 등록 (아래 참조)
```

### 4단계: systemd 서비스 등록 (선택)
```bash
sudo nano /etc/systemd/system/gpt-sovits.service
```

```ini
[Unit]
Description=GPT-SoVITS TTS API Server
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/user/gpt-sovits
ExecStart=/home/user/gpt-sovits/venv/bin/python tts_api_server.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

```bash
# 서비스 활성화 및 시작
sudo systemctl daemon-reload
sudo systemctl enable gpt-sovits
sudo systemctl start gpt-sovits

# 상태 확인
sudo systemctl status gpt-sovits

# 로그 보기
sudo journalctl -u gpt-sovits -f
```

---

## SSL/WSS 설정 (HTTPS 웹페이지용)

HTTPS 웹페이지에서 TTS를 사용하려면 WSS(WebSocket Secure)가 필요합니다.

### 인증서
STT 서버와 같은 인증서 사용 가능 (`key.pem`, `cert.pem`)

### 서버 실행 모드

```bash
# WS만 (기본)
./run-server.sh --mode ws

# WSS만
./run-server.sh --mode wss --ssl-key /path/to/key.pem --ssl-cert /path/to/cert.pem

# WS + WSS 동시 실행
./run-server.sh --mode both --ssl-key /path/to/key.pem --ssl-cert /path/to/cert.pem
```

### 포트

| 모드 | WS (HTTP) | WSS (HTTPS) |
|------|-----------|-------------|
| ws | 9874 | - |
| wss | - | 9875 |
| both | 9874 | 9875 |

---

## 클라이언트 연결

### WebSocket 엔드포인트
```
ws://서버IP:9874/ws/tts
```

### 요청 형식
```json
{
    "text": "안녕하세요",
    "preset_id": "sample",
    "lang": "ko"
}
```

### 응답 순서
1. `{"status": "generating"}` - 생성 시작
2. `{"status": "ready", "sample_rate": 32000, "duration": 1.5, ...}` - 메타데이터
3. `[바이너리 PCM 오디오]` - 16bit, 32kHz, mono

---

## 포트 설정

기본 포트: **9874**

방화벽에서 해당 포트를 열어야 합니다:
```powershell
# Windows 방화벽 규칙 추가 (관리자 권한)
netsh advfirewall firewall add rule name="GPT-SoVITS TTS" dir=in action=allow protocol=TCP localport=9874
```

---

## 모니터링

- 헬스체크: `GET http://서버IP:9874/api/health`
- 통계: `GET http://서버IP:9874/api/stats`
- API 문서: `http://서버IP:9874/docs`
