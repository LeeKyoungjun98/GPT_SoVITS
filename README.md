# GPT-SoVITS TTS API Server

GPT-SoVITS 기반 **Text-to-Speech API 서버**.
WebSocket 스트리밍으로 실시간 음성 합성을 제공합니다.

## 주요 기능

- **WebSocket 실시간 스트리밍** - PCM 16bit, 32kHz 오디오
- **다중 클라이언트** - 최대 50개 동시 연결
- **음성 프리셋** - voiceId로 음성 선택
- **다국어** - 한국어, 일본어, 영어, 중국어 (자동감지)
- **WS/WSS** - HTTP, HTTPS 모두 지원
- **프리셋 캐싱** - 음성 전환 시 빠른 복원 (최대 10개)

---

## 사전 요구사항

- **Python** 3.10 이상
- **NVIDIA GPU** + CUDA 11.8+ (필수)
- **디스크** 약 15GB
- **VRAM** 8GB 이상 권장

---

## 설치 (Windows)

### 1. 저장소 클론

```bash
git clone https://github.com/LeeKyoungjun98/GPT_SoVITS.git
cd GPT_SoVITS
```

### 2. 설치 스크립트 실행

```
install.bat
```

이 스크립트가 자동으로:
1. Python 가상환경(venv) 생성
2. PyTorch + CUDA 11.8 설치
3. requirements.txt 의존성 설치
4. FastAPI, Uvicorn 설치

### 3. 사전학습 모델 다운로드

`GPT_SoVITS/pretrained_models/` 폴더에 다음 모델이 필요합니다:

| 모델 | 경로 |
|------|------|
| GPT-SoVITS v2Pro | `GPT_SoVITS/pretrained_models/v2Pro/` |
| Chinese RoBERTa | `GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/` |
| Chinese HuBERT | `GPT_SoVITS/pretrained_models/chinese-hubert-base/` |

> 모델은 [GPT-SoVITS 원본 저장소](https://github.com/RVC-Boss/GPT-SoVITS)에서 다운로드할 수 있습니다.

### 4. 커스텀 모델 배치 (선택)

학습된 커스텀 모델이 있다면:

```
GPT_weights/       또는  GPT_weights_v2Pro/
└── your_model/
    └── your_model-e15.ckpt

SoVITS_weights/    또는  SoVITS_weights_v2Pro/
└── your_model/
    └── your_model_e8.pth
```

### 5. 영어 TTS 사용 시 (NLTK 리소스)

```bash
venv\Scripts\python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
```

### 6. 서버 실행

```
run-api-server.bat
```

모드 선택 화면:
```
[1] WS only (HTTP) - default
[2] WSS only (HTTPS)
[3] WS + WSS both
```

이후 모델 버전과 커스텀 모델을 선택하면 서버가 시작됩니다.

```
WS:  ws://localhost:9874/ws/tts
API: http://localhost:9874/docs
```

---

## 설치 (Linux)

### 1. 저장소 클론

```bash
git clone https://github.com/YOUR_REPO/GPT_SoVITS.git
cd GPT_SoVITS
```

### 2. 실행 권한 부여

```bash
chmod +x install-linux.sh run-server.sh setup_service.sh
```

### 3. 설치 스크립트 실행

```bash
./install-linux.sh
```

> Python 3.10 이상이 없으면:
> ```bash
> sudo apt update
> sudo apt install -y python3 python3-venv python3-pip
> ```

### 4. CUDA 버전 확인

```bash
nvidia-smi
```

CUDA 12.x인 경우 PyTorch를 재설치:
```bash
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

CUDA 호환 확인:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
`True`가 나오면 성공.

### 5. 사전학습 모델 + 커스텀 모델 배치

Windows와 동일. 모델 파일을 서버에 복사합니다.

### 6. 서버 실행

```bash
./run-server.sh
```

### 7. 백그라운드 실행

**방법 1: screen 사용**
```bash
screen -S tts
./run-server.sh
# 빠져나오기: Ctrl+A → D
# 다시 접속: screen -r tts
```

**방법 2: 자동 재시작 스크립트** (Docker/컨테이너 환경 권장)
```bash
chmod +x run_forever.sh
screen -dmS tts ./run_forever.sh
# 로그 확인: screen -r tts
```
서버가 죽으면 자동으로 재시작됩니다 (최대 100회).

### 8. systemd 서비스 등록 (선택)

```bash
sudo ./setup_service.sh
```

- 부팅 시 자동 시작
- 다운 시 10초 후 자동 재시작

```bash
# 상태 확인
sudo systemctl status gpt-sovits-tts

# 로그 보기
sudo journalctl -u gpt-sovits-tts -f

# 재시작 / 중지
sudo systemctl restart gpt-sovits-tts
sudo systemctl stop gpt-sovits-tts
```

---

## API 사용법

### API 키 인증

서버는 API 키 인증을 지원합니다. 키가 등록되면 모든 API 접근에 인증이 필요합니다.

**키 생성**:
```bash
./venv/bin/python tts_api_server.py --generate-key myapp
```

**키 목록 확인**:
```bash
./venv/bin/python tts_api_server.py --list-keys
```

**인증 비활성화** (개발/테스트용):
```bash
./venv/bin/python tts_api_server.py --no-auth
```

| 방식 | 인증 방법 |
|------|-----------|
| **REST API** | `X-API-Key: gsvtts-xxxx...` 헤더 |
| **WebSocket** | 연결 후 첫 메시지로 `{"api_key": "gsvtts-xxxx..."}` 전송 |

> 키는 `api_keys.json`에 원본 그대로 저장됩니다. 파일을 안전하게 관리하세요.

### WebSocket TTS 스트리밍

**엔드포인트**: `ws://서버:9874/ws/tts`

**요청**:
```json
{
  "text": "안녕하세요",
  "voiceId": "프리셋ID",
  "lang": "auto"
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `text` | string | O | 합성할 텍스트 |
| `voiceId` | string | - | 음성 프리셋 ID |
| `lang` | string | - | 언어 (`auto`, `ko`, `ja`, `en`, `zh`). 기본: `auto` |
| `ref_audio` | string | - | 참조 오디오 경로 (voiceId 없을 때) |
| `ref_text` | string | - | 참조 텍스트 (voiceId 없을 때) |

**응답 순서**:
1. `{"status": "generating"}` - 생성 시작
2. `{"status": "ready", "sample_rate": 32000, ...}` - 메타데이터
3. `[바이너리 PCM 데이터]` - 오디오 (16bit, 32kHz, mono)

### REST API

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| GET | `/api/health` | 헬스체크 |
| GET | `/api/stats` | 서버 통계 |
| GET | `/api/models` | 모델 목록 |
| POST | `/api/model/load` | 모델 로드 |
| GET | `/api/presets` | 프리셋 목록 |
| POST | `/api/presets/add` | 프리셋 추가 |
| POST | `/api/presets/select/{voiceId}` | 프리셋 선택 |
| DELETE | `/api/presets/{voiceId}` | 프리셋 삭제 |
| POST | `/api/reference/set` | 기본 참조 설정 |

API 문서: `http://서버:9874/docs`

---

## 클라이언트 예시

### Python

```python
import asyncio
import websockets
import json

API_KEY = "gsvtts-xxxx..."  # 서버에서 생성한 키

async def tts():
    async with websockets.connect("ws://localhost:9874/ws/tts") as ws:
        # 1) 인증 (첫 메시지)
        await ws.send(json.dumps({"api_key": API_KEY}))
        auth = json.loads(await ws.recv())
        if auth.get("status") != "success":
            print("인증 실패:", auth)
            return

        # 2) TTS 요청
        await ws.send(json.dumps({
            "text": "안녕하세요",
            "voiceId": "your_voice_id"
        }))

        while True:
            msg = await ws.recv()
            if isinstance(msg, str):
                data = json.loads(msg)
                if data.get("status") == "complete":
                    break
            else:
                audio_data = msg  # PCM 오디오 데이터

asyncio.run(tts())
```

### JavaScript

```javascript
const ws = new WebSocket("ws://localhost:9874/ws/tts");

ws.onopen = () => {
  // 1) 인증 (첫 메시지)
  ws.send(JSON.stringify({ api_key: "gsvtts-xxxx..." }));
};

let authenticated = false;
ws.onmessage = (event) => {
  if (typeof event.data === "string") {
    const data = JSON.parse(event.data);

    if (data.type === "auth" && data.status === "success") {
      authenticated = true;
      // 2) 인증 성공 후 TTS 요청
      ws.send(JSON.stringify({
        text: "안녕하세요",
        voiceId: "your_voice_id"
      }));
      return;
    }

    console.log(data);
  } else {
    // PCM 오디오 데이터
    const audioData = event.data;
  }
};
```

### REST API (cURL)

```bash
# 프리셋 목록 조회
curl -H "X-API-Key: gsvtts-xxxx..." http://localhost:9874/api/presets

# 프리셋 선택
curl -X POST -H "X-API-Key: gsvtts-xxxx..." http://localhost:9874/api/presets/select/your_voice_id
```

---

## 프리셋 설정

`presets/presets.json`:

```json
{
  "presets": {
    "voice_id_here": {
      "name": "화자 이름",
      "langs": {
        "ko": {
          "audio": "speaker_ko.mp3",
          "text": "한국어 참조 텍스트"
        },
        "ja": {
          "audio": "speaker_ja.mp3",
          "text": "日本語の参照テキスト"
        },
        "en": {
          "audio": "speaker_en.mp3",
          "text": "English reference text."
        }
      }
    }
  }
}
```

- 참조 오디오 파일은 `presets/` 폴더에 배치
- `lang: "auto"` 시 텍스트 언어를 감지하여 해당 언어의 참조 오디오 자동 선택
- 요청 언어의 오디오가 없으면 한국어(ko)로 폴백

---

## 폴더 구조

```
GPT_SoVITS/
├── tts_api_server.py          # API 서버
├── test_tts_pipeline.py       # 테스트 클라이언트
├── run-api-server.bat         # Windows 실행
├── run-server.sh              # Linux 실행
├── run_forever.sh             # 자동 재시작 실행 (Linux)
├── install.bat                # Windows 설치
├── install-linux.sh           # Linux 설치
├── setup_service.sh           # systemd 서비스 설정
├── api_keys.json              # API 키 저장 (자동 생성)
├── requirements.txt           # Python 의존성
├── presets/                   # 음성 프리셋
│   ├── presets.json
│   └── *.mp3, *.wav           # 참조 오디오
├── GPT_SoVITS/                # 코어 코드
│   ├── TTS_infer_pack/
│   ├── module/
│   ├── text/
│   └── pretrained_models/     # 사전학습 모델
├── GPT_weights*/              # 커스텀 GPT 모델
├── SoVITS_weights*/           # 커스텀 SoVITS 모델
└── tools/
```

---

## 포트

| 모드 | 프로토콜 | 포트 | 용도 |
|------|----------|------|------|
| ws | HTTP | 9874 | 일반 연결 |
| wss | HTTPS | 9875 | 보안 연결 |

---

## TTS 성능 설정

`tts_api_server.py` 상단에서 조절 가능:

```python
TTS_TOP_K = 5           # 낮을수록 빠름 (1-100)
TTS_TEMPERATURE = 0.8   # 낮을수록 빠름 (0.1-1.0)
TTS_BATCH_SIZE = 40     # 높을수록 빠름 (VRAM 8GB+)
TTS_CUT_METHOD = "cut0" # cut0: 분할없음(빠름), cut1: 문장단위
TTS_SPEED = 1.0         # 재생 속도
```

---

## 문제 해결

### CUDA 관련
```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"

# CUDA 12.x에서 PyTorch 재설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 포트 충돌
```bash
# 사용 중인 포트 확인
sudo lsof -i :9874  # Linux
netstat -ano | findstr :9874  # Windows
```

### 영어 TTS 오류 (NLTK)
```bash
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
```

### pyopenjtalk 빌드 오류 (Windows)
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) 설치
- "C++를 사용한 데스크톱 개발" 워크로드 선택

### 모델 로드 실패
- `GPT_SoVITS/pretrained_models/` 폴더에 모델이 있는지 확인
- 커스텀 모델은 `GPT_weights*/`, `SoVITS_weights*/`에 배치
