"""
GPT-SoVITS TTS API Server
REST API로 텍스트를 받아 스트리밍 음성을 반환하는 서버

[프로덕션 기능]
- 동시 처리 제한 (세마포어)
- 최대 연결 수 제한
- 연결 상태 모니터링
- 요청 큐 시스템
"""

import os
import sys
import io
import wave
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Optional, Generator, Dict
from contextlib import asynccontextmanager
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# 경로 설정
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir, "GPT_SoVITS"))

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import threading
import queue
import secrets

# 언어 감지
try:
    import fast_langdetect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    logger.warning("fast_langdetect 없음 - 자동 언어 감지 불가")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
is_half = True if torch.cuda.is_available() else False

# ============================================
# 프로덕션 설정
# ============================================
# ============================================
# 프로덕션 설정 (GPU에 따라 자동 조절)
# ============================================
MAX_CONNECTIONS = 50              # 최대 동시 WebSocket 연결 수
REQUEST_TIMEOUT = 60              # 요청 타임아웃 (초)

# GPU별 동시 TTS 수 자동 설정
def _detect_gpu_tier():
    if not torch.cuda.is_available():
        return 1, 40, 2  # CPU: 1 concurrent, batch 40, 2 workers
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    name = torch.cuda.get_device_properties(0).name
    if vram_gb >= 70:  # H100 80GB, A100 80GB
        return 4, 80, 6
    elif vram_gb >= 30:  # A100 40GB, V100 32GB
        return 2, 60, 4
    elif vram_gb >= 16:  # A10, T4 16GB
        return 2, 50, 3
    else:  # 8GB
        return 1, 40, 2

MAX_CONCURRENT_TTS, TTS_BATCH_SIZE, _TTS_WORKERS = _detect_gpu_tier()
logger.info(f"GPU 감지: {torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'} "
            f"-> 동시TTS={MAX_CONCURRENT_TTS}, 배치={TTS_BATCH_SIZE}, 워커={_TTS_WORKERS}")

# ============================================
# API 키 인증
# ============================================
API_KEY_FILE = os.path.join(os.path.dirname(__file__), "api_keys.json")
AUTH_ENABLED = True

class APIKeyManager:
    """API 키 관리자 (원본 키 저장 방식)"""
    
    def __init__(self, keys_file: str):
        self.keys_file = keys_file
        self.api_keys: Dict[str, dict] = {}
        self._load_keys()
    
    def _load_keys(self):
        if os.path.exists(self.keys_file):
            try:
                with open(self.keys_file, "r", encoding="utf-8") as f:
                    self.api_keys = json.load(f)
                logger.info(f"API 키 {len(self.api_keys)}개 로드됨")
            except Exception as e:
                logger.error(f"API 키 파일 로드 실패: {e}")
                self.api_keys = {}
        else:
            self.api_keys = {}
            self._save_keys()
    
    def _save_keys(self):
        try:
            with open(self.keys_file, "w", encoding="utf-8") as f:
                json.dump(self.api_keys, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"API 키 파일 저장 실패: {e}")
    
    def generate_key(self, name: str = "") -> str:
        key = f"gsvtts-{secrets.token_hex(24)}"
        self.api_keys[key] = {
            "name": name or f"key_{len(self.api_keys) + 1}",
            "created_at": datetime.now().isoformat(),
            "last_used": None,
            "request_count": 0
        }
        self._save_keys()
        return key
    
    def validate_key(self, key: str) -> bool:
        if not key or key not in self.api_keys:
            return False
        self.api_keys[key]["last_used"] = datetime.now().isoformat()
        self.api_keys[key]["request_count"] += 1
        if self.api_keys[key]["request_count"] % 100 == 0:
            self._save_keys()
        return True
    
    def revoke_key(self, key: str) -> bool:
        if key in self.api_keys:
            del self.api_keys[key]
            self._save_keys()
            return True
        return False
    
    def list_keys(self) -> list:
        result = []
        for key, info in self.api_keys.items():
            result.append({
                "key_preview": f"{key[:12]}...{key[-4:]}",
                "name": info["name"],
                "created_at": info["created_at"],
                "last_used": info["last_used"],
                "request_count": info["request_count"]
            })
        return result

api_key_manager = APIKeyManager(API_KEY_FILE)

def verify_api_key(key: str) -> bool:
    if not AUTH_ENABLED:
        return True
    return api_key_manager.validate_key(key)

async def get_api_key_from_header(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """REST API용 API 키 검증 (X-API-Key 헤더)"""
    if not AUTH_ENABLED:
        return "no-auth"
    
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API 키가 필요합니다. 'X-API-Key' 헤더를 포함해주세요.")
    
    if not api_key_manager.validate_key(x_api_key):
        raise HTTPException(status_code=403, detail="유효하지 않은 API 키입니다.")
    
    return x_api_key

# ============================================
# TTS 속도 최적화 설정
# ============================================
TTS_TOP_K = 5                     # 낮을수록 빠름 (기본: 5, 범위: 1-100)
TTS_TOP_P = 1.0                   # Top-P 샘플링 (기본: 1.0)
TTS_TEMPERATURE = 0.8             # 낮을수록 빠름 (기본: 0.8, 범위: 0.1-1.0)
# TTS_BATCH_SIZE는 위에서 GPU별 자동 설정됨
TTS_CUT_METHOD = "cut0"           # cut0:분할없음(빠름), cut1:문장, cut5:4문장
TTS_SPEED = 1.0                   # 재생 속도 (1.0 = 기본)

# 동시 처리 제한 세마포어
tts_semaphore: Optional[asyncio.Semaphore] = None

# TTS 작업용 스레드풀
tts_executor = ThreadPoolExecutor(max_workers=_TTS_WORKERS, thread_name_prefix="tts")

# TTS 엔진 (전역)
tts_engine = None
current_version = None
current_gpt_path = None
current_sovits_path = None


# ============================================
# 프리셋별 캐시 관리자
# ============================================
class PresetCacheManager:
    """
    프리셋별 캐시 관리
    GPT-SoVITS의 prompt_cache를 프리셋별로 저장/복원하여
    여러 프리셋을 번갈아 사용해도 재로드 없이 빠르게 전환
    """
    
    def __init__(self, max_cached_presets: int = 10):
        self.caches: Dict[str, dict] = {}  # preset_id -> prompt_cache 복사본
        self.max_cached = max_cached_presets
        self.current_preset_id: Optional[str] = None
        self._lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def save_current_cache(self, preset_id: str):
        """현재 TTS 엔진의 캐시를 저장"""
        if tts_engine is None:
            return
        
        with self._lock:
            # 깊은 복사로 캐시 저장
            import copy
            cache_copy = {}
            for key, value in tts_engine.prompt_cache.items():
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    cache_copy[key] = value.clone() if isinstance(value, torch.Tensor) else value.copy()
                elif isinstance(value, list):
                    cache_copy[key] = copy.deepcopy(value)
                else:
                    cache_copy[key] = value
            
            self.caches[preset_id] = cache_copy
            self.current_preset_id = preset_id
            
            # 최대 개수 초과 시 가장 오래된 것 제거
            if len(self.caches) > self.max_cached:
                oldest = next(iter(self.caches))
                if oldest != preset_id:
                    del self.caches[oldest]
                    logger.info(f"[캐시] 오래된 프리셋 제거: {oldest}")
    
    def restore_cache(self, preset_id: str) -> bool:
        """저장된 캐시를 TTS 엔진에 복원"""
        if tts_engine is None:
            return False
        
        with self._lock:
            if preset_id not in self.caches:
                self.cache_misses += 1
                return False
            
            # 현재 캐시 먼저 저장 (있다면)
            if self.current_preset_id and self.current_preset_id != preset_id:
                self._save_current_internal()
            
            # 캐시 복원
            import copy
            saved_cache = self.caches[preset_id]
            for key, value in saved_cache.items():
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    tts_engine.prompt_cache[key] = value.clone() if isinstance(value, torch.Tensor) else value.copy()
                elif isinstance(value, list):
                    tts_engine.prompt_cache[key] = copy.deepcopy(value)
                else:
                    tts_engine.prompt_cache[key] = value
            
            self.current_preset_id = preset_id
            self.cache_hits += 1
            return True
    
    def _save_current_internal(self):
        """내부용: 락 없이 현재 캐시 저장"""
        if self.current_preset_id and tts_engine:
            import copy
            cache_copy = {}
            for key, value in tts_engine.prompt_cache.items():
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    cache_copy[key] = value.clone() if isinstance(value, torch.Tensor) else value.copy()
                elif isinstance(value, list):
                    cache_copy[key] = copy.deepcopy(value)
                else:
                    cache_copy[key] = value
            self.caches[self.current_preset_id] = cache_copy
    
    def is_current(self, preset_id: str) -> bool:
        """현재 로드된 프리셋인지 확인"""
        return self.current_preset_id == preset_id
    
    def has_cache(self, preset_id: str) -> bool:
        """캐시가 있는지 확인"""
        return preset_id in self.caches
    
    def get_stats(self) -> dict:
        """캐시 통계"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            "cached_presets": list(self.caches.keys()),
            "count": len(self.caches),
            "max": self.max_cached,
            "current": self.current_preset_id,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }
    
    def clear(self):
        """모든 캐시 초기화"""
        with self._lock:
            self.caches.clear()
            self.current_preset_id = None


# 프리셋 캐시 매니저 인스턴스
preset_cache_manager = PresetCacheManager(max_cached_presets=10)


# ============================================
# 연결 관리자
# ============================================
class ConnectionManager:
    """WebSocket 연결 관리자"""
    
    def __init__(self):
        self.active_connections: Dict[int, dict] = {}
        self.total_connections_served = 0
        self.total_tts_requests = 0
        self.server_start_time = datetime.now()
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, client_id: int) -> bool:
        """새 연결 등록 (최대 연결 수 체크)"""
        async with self._lock:
            if len(self.active_connections) >= MAX_CONNECTIONS:
                return False
            
            self.active_connections[client_id] = {
                "websocket": websocket,
                "connected_at": datetime.now(),
                "last_activity": datetime.now(),
                "tts_count": 0,
                "voiceId": None,
                "client_info": str(websocket.client) if websocket.client else "unknown"
            }
            self.total_connections_served += 1
            return True
    
    async def disconnect(self, client_id: int):
        """연결 해제"""
        async with self._lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
    
    async def update_activity(self, client_id: int):
        """활동 시간 업데이트"""
        if client_id in self.active_connections:
            self.active_connections[client_id]["last_activity"] = datetime.now()
    
    async def increment_tts(self, client_id: int):
        """TTS 카운트 증가"""
        async with self._lock:
            if client_id in self.active_connections:
                self.active_connections[client_id]["tts_count"] += 1
            self.total_tts_requests += 1
    
    def set_client_voice(self, client_id: int, voice_id: str):
        """클라이언트별 음성 설정"""
        if client_id in self.active_connections:
            self.active_connections[client_id]["voiceId"] = voice_id
    
    def get_client_voice(self, client_id: int) -> Optional[str]:
        """클라이언트의 현재 음성 가져오기"""
        if client_id in self.active_connections:
            return self.active_connections[client_id].get("voiceId")
        return None
    
    def get_stats(self) -> dict:
        """서버 통계 반환"""
        uptime = datetime.now() - self.server_start_time
        pending = MAX_CONCURRENT_TTS - tts_semaphore._value if tts_semaphore else 0
        return {
            "active_connections": len(self.active_connections),
            "max_connections": MAX_CONNECTIONS,
            "total_connections_served": self.total_connections_served,
            "total_tts_requests": self.total_tts_requests,
            "active_tts": pending,
            "max_concurrent_tts": MAX_CONCURRENT_TTS,
            "uptime_seconds": int(uptime.total_seconds()),
            "uptime_formatted": str(uptime).split('.')[0]
        }
    
    def get_connection_details(self) -> list:
        """모든 연결 상세 정보"""
        details = []
        for client_id, info in self.active_connections.items():
            connected_duration = datetime.now() - info["connected_at"]
            details.append({
                "client_id": client_id,
                "client_info": info["client_info"],
                "connected_duration": str(connected_duration).split('.')[0],
                "tts_count": info["tts_count"],
                "voiceId": info["voiceId"]
            })
        return details


# 연결 관리자 인스턴스
connection_manager = ConnectionManager()

# 언어 옵션
LANGUAGES = {
    "ko": "all_ko",
    "korean": "all_ko",
    "zh": "all_zh",
    "chinese": "all_zh",
    "en": "en",
    "english": "en",
    "ja": "all_ja",
    "japanese": "all_ja",
    "yue": "all_yue",
    "cantonese": "all_yue",
    "ko_en": "ko",
    "zh_en": "zh",
    "ja_en": "ja",
    "auto": "auto",
}

# 모델 버전
MODEL_VERSIONS = ["v1", "v2", "v3", "v4", "v2Pro"]

# 시작 시 로드할 모델 설정 (main에서 변경됨)
STARTUP_MODEL = {
    "version": "v2Pro",
    "custom_gpt": "",
    "custom_sovits": ""
}

# 기본 참조 오디오 설정 (서버에서 미리 로드)
DEFAULT_REFERENCE = {
    "audio_path": "",
    "text": "",
    "lang": "ko"
}

# 프리셋 저장 경로
PRESETS_DIR = os.path.join(now_dir, "presets")
PRESETS_FILE = os.path.join(PRESETS_DIR, "presets.json")

# 음성 프리셋 (ID로 선택 가능)
VOICE_PRESETS = {}

# 현재 선택된 프리셋 ID
CURRENT_PRESET_ID = None


def resolve_audio_path(audio_path: str) -> str:
    """오디오 경로를 절대 경로로 변환
    
    - 절대 경로면 그대로 반환
    - 상대 경로면 PRESETS_DIR 기준으로 변환
    """
    if os.path.isabs(audio_path):
        return audio_path
    else:
        return os.path.join(PRESETS_DIR, audio_path)


def detect_language(text: str) -> str:
    """텍스트의 언어를 감지하여 반환
    
    Returns:
        언어 코드 (ko, ja, en, zh 등)
    """
    if not HAS_LANGDETECT or not text:
        return "ko"  # 기본값
    
    try:
        result = fast_langdetect.detect(text)
        
        # fast_langdetect 반환 형식 처리 (버전에 따라 다름)
        if isinstance(result, list) and len(result) > 0:
            # [{'lang': 'ja', 'score': 0.99}] 형식
            detected = result[0].get("lang", "ko") if isinstance(result[0], dict) else str(result[0])
        elif isinstance(result, dict):
            detected = result.get("lang", "ko")
        elif isinstance(result, str):
            detected = result
        elif hasattr(result, "lang"):
            detected = result.lang
        else:
            detected = str(result)
        
        # 언어 코드 정규화
        lang_map = {
            "ko": "ko",
            "ja": "ja",
            "en": "en",
            "zh": "zh",
            "zh-cn": "zh",
            "zh-tw": "zh",
        }
        normalized = lang_map.get(detected.lower(), "ko")
        logger.info(f"[언어 감지] '{text[:20]}...' -> {detected} -> {normalized}")
        return normalized
    except Exception as e:
        logger.warning(f"[언어 감지 실패] {e}")
        return "ko"


def get_preset_for_lang(preset: dict, lang: str) -> tuple:
    """프리셋에서 언어에 맞는 오디오와 텍스트 반환
    
    지원하는 프리셋 구조:
    1. 단일 언어 (기존): {"audio": "...", "text": "...", "lang": "ko"}
    2. 다국어: {"langs": {"ko": {"audio": "...", "text": "..."}, "ja": {...}}}
    
    Returns:
        (audio_path, ref_text, actual_lang)
    """
    # 언어 코드 매핑 (ja <-> jp 등)
    lang_aliases = {
        "ja": "jp",
        "jp": "ja",
        "japanese": "jp",
        "korean": "ko",
        "english": "en",
        "chinese": "zh",
    }
    
    # 다국어 구조인 경우
    if "langs" in preset:
        langs = preset["langs"]
        available_langs = list(langs.keys())
        logger.info(f"[프리셋 언어 선택] 요청: {lang}, 사용 가능: {available_langs}")
        
        # 요청한 언어가 있으면 사용
        if lang in langs and langs[lang].get("audio"):
            lang_data = langs[lang]
            logger.info(f"[프리셋 언어 선택] 직접 매칭: {lang}")
            return (
                resolve_audio_path(lang_data["audio"]),
                lang_data.get("text", ""),
                lang
            )
        
        # 언어 별칭으로 시도 (ja -> jp 등)
        alias_lang = lang_aliases.get(lang)
        logger.info(f"[프리셋 언어 선택] 별칭 시도: {lang} -> {alias_lang}")
        if alias_lang and alias_lang in langs and langs[alias_lang].get("audio"):
            lang_data = langs[alias_lang]
            logger.info(f"[프리셋 언어 선택] 별칭 매칭: {alias_lang}")
            return (
                resolve_audio_path(lang_data["audio"]),
                lang_data.get("text", ""),
                alias_lang
            )
        
        # 없으면 한국어로 폴백
        logger.info(f"[프리셋 언어 선택] 폴백: ko")
        if "ko" in langs and langs["ko"].get("audio"):
            lang_data = langs["ko"]
            return (
                resolve_audio_path(lang_data["audio"]),
                lang_data.get("text", ""),
                "ko"
            )
        
        # 한국어도 없으면 첫 번째 언어 사용
        for fallback_lang, lang_data in langs.items():
            if lang_data.get("audio"):
                return (
                    resolve_audio_path(lang_data["audio"]),
                    lang_data.get("text", ""),
                    fallback_lang
                )
        
        return (None, "", lang)
    
    # 단일 언어 구조 (기존 호환)
    else:
        return (
            resolve_audio_path(preset.get("audio", "")),
            preset.get("text", ""),
            preset.get("lang", "ko")
        )


def load_presets():
    """프리셋 파일에서 로드"""
    global VOICE_PRESETS, CURRENT_PRESET_ID
    
    # 폴더 없으면 생성
    if not os.path.exists(PRESETS_DIR):
        os.makedirs(PRESETS_DIR)
    
    # 파일 있으면 로드
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                VOICE_PRESETS = data.get("presets", {})
                last_selected = data.get("last_selected")
                
                print(f"[프리셋] {len(VOICE_PRESETS)}개 로드됨")
                for pid, preset in VOICE_PRESETS.items():
                    marker = " *" if pid == last_selected else ""
                    print(f"  - {pid}: {preset.get('name', pid)}{marker}")
                
                # 마지막 선택된 프리셋 복원 (모델 로드 후 캐싱은 별도)
                if last_selected and last_selected in VOICE_PRESETS:
                    CURRENT_PRESET_ID = last_selected
                    preset = VOICE_PRESETS[last_selected]
                    audio_path, ref_text, ref_lang = get_preset_for_lang(preset, "ko")
                    DEFAULT_REFERENCE["audio_path"] = audio_path
                    DEFAULT_REFERENCE["text"] = ref_text
                    DEFAULT_REFERENCE["lang"] = ref_lang
                    
        except Exception as e:
            print(f"[프리셋] 로드 실패: {e}")
            VOICE_PRESETS = {}
    else:
        print(f"[프리셋] 파일 없음, 새로 생성됩니다")


def save_presets():
    """프리셋을 파일로 저장"""
    try:
        # 폴더 없으면 생성
        if not os.path.exists(PRESETS_DIR):
            os.makedirs(PRESETS_DIR)
        
        with open(PRESETS_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                "presets": VOICE_PRESETS,
                "last_selected": CURRENT_PRESET_ID
            }, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[프리셋] 저장 실패: {e}")
        return False


# ============================================================
# Pydantic 모델
# ============================================================

class TTSRequest(BaseModel):
    """TTS 합성 요청"""
    text: str = Field(..., description="합성할 텍스트")
    text_lang: str = Field("ko", description="텍스트 언어 (ko, en, ja, zh, auto)")
    ref_audio_path: str = Field(..., description="참조 오디오 파일 경로")
    ref_text: str = Field("", description="참조 오디오의 텍스트 (v3/v4 필수)")
    ref_lang: str = Field("ko", description="참조 오디오 언어")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="재생 속도 (0.5~2.0)")
    top_k: int = Field(5, ge=1, le=100, description="Top-K 샘플링 (낮을수록 빠름)")
    top_p: float = Field(1.0, ge=0.1, le=1.0, description="Top-P 샘플링")
    temperature: float = Field(0.8, ge=0.1, le=1.0, description="Temperature (낮을수록 빠름)")
    streaming: bool = Field(False, description="스트리밍 모드 (True: 문장별 반환, False: 빠른 일괄 생성)")
    format: str = Field("wav", description="출력 포맷 (wav, pcm)")
    cut_method: str = Field("cut0", description="텍스트 분할 (cut0:없음-빠름, cut1:문장, cut5:4문장)")
    batch_size: int = Field(40, ge=1, le=200, description="배치 크기 (높을수록 빠름, VRAM 더 사용)")
    parallel_infer: bool = Field(True, description="병렬 추론 (True: 빠름)")


class ModelLoadRequest(BaseModel):
    """모델 로드 요청"""
    version: str = Field("v2", description="모델 버전 (v1, v2, v3, v4, v2Pro)")
    custom_gpt: str = Field("", description="커스텀 GPT 모델 경로")
    custom_sovits: str = Field("", description="커스텀 SoVITS 모델 경로")


class TTSResponse(BaseModel):
    """TTS 응답 메타데이터"""
    success: bool
    message: str
    audio_duration: Optional[float] = None
    elapsed_time: Optional[float] = None
    rtf: Optional[float] = None


# ============================================================
# TTS 핵심 함수
# ============================================================

def get_tts_config(version: str):
    """버전에 맞는 TTS 설정 생성"""
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config
    
    config = TTS_Config()
    config.device = torch.device(device)
    config.is_half = is_half
    
    if version in config.default_configs:
        version_config = config.default_configs[version]
        config.version = version
        config.t2s_weights_path = version_config["t2s_weights_path"]
        config.vits_weights_path = version_config["vits_weights_path"]
        config.bert_base_path = version_config["bert_base_path"]
        config.cnhuhbert_base_path = version_config["cnhuhbert_base_path"]
    
    return config


def load_tts_model(version: str, custom_gpt: str = "", custom_sovits: str = ""):
    """모델 로드"""
    global tts_engine, current_version, current_gpt_path, current_sovits_path
    
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS
    
    # 이미 같은 모델이 로드되어 있으면 스킵
    if (tts_engine is not None and 
        current_version == version and 
        current_gpt_path == custom_gpt and 
        current_sovits_path == custom_sovits):
        return True, "모델이 이미 로드되어 있습니다."
    
    try:
        # 기존 모델 정리
        if tts_engine is not None:
            del tts_engine
            tts_engine = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        config = get_tts_config(version)
        
        # 커스텀 모델 경로 적용
        if custom_gpt and os.path.exists(custom_gpt):
            config.t2s_weights_path = custom_gpt
        
        if custom_sovits and os.path.exists(custom_sovits):
            config.vits_weights_path = custom_sovits
        
        tts_engine = TTS(config)
        current_version = version
        current_gpt_path = custom_gpt
        current_sovits_path = custom_sovits
        
        return True, f"모델 로드 완료: {version}"
    
    except Exception as e:
        return False, f"모델 로드 실패: {str(e)}"


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """numpy 배열을 WAV 바이트로 변환"""
    # int16으로 변환
    if audio.dtype == np.float32 or audio.dtype == np.float64:
        audio = (audio * 32767).astype(np.int16)
    elif audio.dtype != np.int16:
        audio = audio.astype(np.int16)
    
    # WAV 파일 생성
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    
    return buffer.getvalue()


def numpy_to_pcm_bytes(audio: np.ndarray) -> bytes:
    """numpy 배열을 PCM 바이트로 변환"""
    if audio.dtype == np.float32 or audio.dtype == np.float64:
        audio = (audio * 32767).astype(np.int16)
    elif audio.dtype != np.int16:
        audio = audio.astype(np.int16)
    
    return audio.tobytes()


def generate_audio_stream(request: TTSRequest) -> Generator[bytes, None, None]:
    """스트리밍 오디오 생성 (generator)"""
    global tts_engine
    
    if tts_engine is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다. /api/model/load 호출 필요")
    
    # 언어 코드 변환
    text_lang_code = LANGUAGES.get(request.text_lang.lower(), "auto")
    ref_lang_code = LANGUAGES.get(request.ref_lang.lower(), "auto")
    
    # TTS 입력 설정
    inputs = {
        "text": request.text,
        "text_lang": text_lang_code,
        "ref_audio_path": request.ref_audio_path,
        "prompt_text": request.ref_text if request.ref_text.strip() else "",
        "prompt_lang": ref_lang_code,
        "top_k": request.top_k,
        "top_p": request.top_p,
        "temperature": request.temperature,
        "speed_factor": request.speed,
        "text_split_method": request.cut_method,
        "batch_size": request.batch_size,
        "return_fragment": request.streaming,  # 스트리밍이면 fragment 모드
        "parallel_infer": request.parallel_infer,
        "split_bucket": request.parallel_infer,  # 병렬 추론 시 버킷 분할
        "sample_steps": 32 if current_version == "v3" else 8,
        "seed": -1,
    }
    
    # 합성 실행 (상세 타이밍 로그)
    start = time.time()
    chunk_idx = 0
    
    def log(msg):
        elapsed = time.time() - start
        print(f"[TTS +{elapsed:5.2f}s] {msg}")
    
    log(f"시작: '{request.text[:50]}...' (lang={text_lang_code})")
    log(f"참조: {os.path.basename(request.ref_audio_path)}")
    log(f"설정: top_k={request.top_k}, cut={request.cut_method}")
    
    for sr, audio in tts_engine.run(inputs):
        chunk_idx += 1
        audio_len = len(audio) / sr
        log(f"청크 #{chunk_idx} 생성됨: {audio_len:.2f}초 오디오 ({len(audio)} samples)")
        
        if request.format == "wav":
            yield numpy_to_wav_bytes(audio, sr)
        else:
            yield numpy_to_pcm_bytes(audio)
        
        log(f"청크 #{chunk_idx} 전송됨")
    
    log(f"완료: 총 {chunk_idx}개 청크")


def generate_full_audio(request: TTSRequest) -> tuple:
    """전체 오디오 생성 (non-streaming)"""
    global tts_engine
    
    if tts_engine is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")
    
    # 언어 코드 변환
    text_lang_code = LANGUAGES.get(request.text_lang.lower(), "auto")
    ref_lang_code = LANGUAGES.get(request.ref_lang.lower(), "auto")
    
    inputs = {
        "text": request.text,
        "text_lang": text_lang_code,
        "ref_audio_path": request.ref_audio_path,
        "prompt_text": request.ref_text if request.ref_text.strip() else "",
        "prompt_lang": ref_lang_code,
        "top_k": request.top_k,
        "top_p": request.top_p,
        "temperature": request.temperature,
        "speed_factor": request.speed,
        "text_split_method": request.cut_method,
        "batch_size": request.batch_size,
        "return_fragment": False,
        "parallel_infer": request.parallel_infer,
        "split_bucket": request.parallel_infer,
        "sample_steps": 32 if current_version == "v3" else 8,
        "seed": -1,
    }
    
    start_time = time.time()
    all_audio = []
    sample_rate = None
    
    for sr, audio in tts_engine.run(inputs):
        sample_rate = sr
        all_audio.append(audio)
    
    elapsed_time = time.time() - start_time
    
    if all_audio:
        final_audio = np.concatenate(all_audio) if len(all_audio) > 1 else all_audio[0]
        audio_duration = len(final_audio) / sample_rate
        
        if request.format == "wav":
            audio_bytes = numpy_to_wav_bytes(final_audio, sample_rate)
        else:
            audio_bytes = numpy_to_pcm_bytes(final_audio)
        
        return audio_bytes, sample_rate, audio_duration, elapsed_time
    
    raise HTTPException(status_code=500, detail="오디오 생성 실패")


def get_available_models():
    """사용 가능한 커스텀 모델 목록 조회"""
    gpt_models = []
    sovits_models = []
    
    # GPT 모델 검색
    for folder in ["GPT_weights", "GPT_weights_v2"]:
        if os.path.exists(folder):
            for root, dirs, files in os.walk(folder):
                for f in files:
                    if f.endswith(".ckpt"):
                        gpt_models.append(os.path.join(root, f))
    
    # SoVITS 모델 검색
    for folder in ["SoVITS_weights", "SoVITS_weights_v2"]:
        if os.path.exists(folder):
            for root, dirs, files in os.walk(folder):
                for f in files:
                    if f.endswith(".pth"):
                        sovits_models.append(os.path.join(root, f))
    
    return gpt_models, sovits_models


def get_reference_audios():
    """참조 오디오 목록"""
    ref_audios = {}
    
    dataset_folders = [
        "dataset/starrail/all_characters",
        "dataset",
        "output/slicer_opt",
    ]
    
    for base_folder in dataset_folders:
        if not os.path.exists(base_folder):
            continue
        
        for root, dirs, files in os.walk(base_folder):
            wav_files = [f for f in files if f.endswith(('.wav', '.mp3', '.flac'))]
            if wav_files:
                char_name = os.path.basename(root)
                if char_name not in ref_audios:
                    ref_audios[char_name] = []
                
                for f in wav_files[:100]:  # 캐릭터당 최대 100개
                    ref_audios[char_name].append({
                        "name": f,
                        "path": os.path.join(root, f)
                    })
    
    return ref_audios


# ============================================================
# FastAPI 앱
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행"""
    global tts_semaphore
    
    print("=" * 60)
    print("  GPT-SoVITS TTS API Server")
    print(f"  Device: {device}")
    print(f"  Half Precision: {is_half}")
    print("=" * 60)
    
    # 세마포어 초기화 (동시 TTS 제한)
    tts_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TTS)
    logger.info(f"동시 처리 세마포어 초기화: {MAX_CONCURRENT_TTS}개")
    logger.info(f"최대 연결 수: {MAX_CONNECTIONS}")
    
    # 프리셋 로드
    load_presets()
    
    # 시작 시 모델 자동 로드
    if STARTUP_MODEL.get("version"):
        version = STARTUP_MODEL["version"]
        custom_gpt = STARTUP_MODEL.get("custom_gpt", "")
        custom_sovits = STARTUP_MODEL.get("custom_sovits", "")
        
        print(f"\n[자동 로드] {version} 모델 로딩 중...")
        if custom_gpt:
            print(f"  GPT: {os.path.basename(custom_gpt)}")
        if custom_sovits:
            print(f"  SoVITS: {os.path.basename(custom_sovits)}")
        
        success, message = load_tts_model(version, custom_gpt, custom_sovits)
        if success:
            print(f"[자동 로드] 완료: {message}")
            
            # 마지막 선택된 프리셋 자동 캐싱
            if CURRENT_PRESET_ID and CURRENT_PRESET_ID in VOICE_PRESETS:
                preset = VOICE_PRESETS[CURRENT_PRESET_ID]
                audio_path, ref_text, ref_lang = get_preset_for_lang(preset, "ko")
                print(f"\n[프리셋 캐싱] {CURRENT_PRESET_ID}: {preset.get('name', '')} ({ref_lang})")
                try:
                    tts_engine.set_ref_audio(audio_path)
                    if ref_text:
                        lang_code = LANGUAGES.get(ref_lang.lower(), "auto")
                        phones, bert_features, norm_text = tts_engine.text_preprocessor.segment_and_extract_feature_for_text(
                            ref_text, lang_code, tts_engine.configs.version
                        )
                        tts_engine.prompt_cache["prompt_text"] = ref_text
                        tts_engine.prompt_cache["prompt_lang"] = lang_code
                        tts_engine.prompt_cache["phones"] = phones
                        tts_engine.prompt_cache["bert_features"] = bert_features
                        tts_engine.prompt_cache["norm_text"] = norm_text
                    print(f"[프리셋 캐싱] 완료")
                except Exception as e:
                    print(f"[프리셋 캐싱] 실패: {e}")
        else:
            print(f"[자동 로드] 실패: {message}")
    else:
        print("\n[자동 로드] 비활성화됨 - API로 모델을 로드하세요")
    print()
    
    yield
    # 종료 시 정리
    if tts_engine is not None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

app = FastAPI(
    title="GPT-SoVITS TTS API",
    description="텍스트를 음성으로 변환하는 REST API (스트리밍 지원)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 키 인증 미들웨어
PUBLIC_PATHS = {"/", "/docs", "/redoc", "/openapi.json", "/api/health"}

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """API 키 인증 미들웨어 - X-API-Key 헤더 방식"""
    if not AUTH_ENABLED:
        return await call_next(request)
    
    if len(api_key_manager.api_keys) == 0:
        return await call_next(request)
    
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)
    
    api_key = request.headers.get("x-api-key", "")
    if not api_key_manager.validate_key(api_key):
        return JSONResponse(status_code=403, content={"detail": "유효하지 않은 API 키입니다. 'X-API-Key' 헤더를 포함해주세요."})
    
    return await call_next(request)


# ============================================================
# API 엔드포인트
# ============================================================

@app.get("/")
async def root():
    """API 상태 확인"""
    return {
        "status": "running",
        "service": "GPT-SoVITS TTS API",
        "model_loaded": tts_engine is not None,
        "current_version": current_version,
        "device": device
    }


@app.get("/api/health")
async def health_check():
    """헬스 체크"""
    stats = connection_manager.get_stats()
    
    # GPU 메모리 상태
    gpu_info = {}
    if device == "cuda":
        try:
            gpu_info = {
                "gpu_memory_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 1),
                "gpu_name": torch.cuda.get_device_name(0)
            }
        except:
            pass
    
    return {
        "status": "healthy",
        "model_ready": tts_engine is not None,
        "model_version": current_version,
        "device": device,
        "connections": f"{stats['active_connections']}/{MAX_CONNECTIONS}",
        "uptime": stats["uptime_formatted"],
        **gpu_info
    }


@app.get("/api/stats")
async def get_server_stats():
    """상세 서버 통계"""
    stats = connection_manager.get_stats()
    connections = connection_manager.get_connection_details()
    cache_stats = preset_cache_manager.get_stats()
    
    return {
        **stats,
        "connections_detail": connections,
        "presets_count": len(VOICE_PRESETS),
        "current_preset": CURRENT_PRESET_ID,
        "preset_cache": cache_stats
    }


@app.post("/api/model/load")
async def load_model(request: ModelLoadRequest):
    """모델 로드"""
    if request.version not in MODEL_VERSIONS:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 버전: {request.version}")
    
    success, message = load_tts_model(request.version, request.custom_gpt, request.custom_sovits)
    
    if success:
        return {"success": True, "message": message, "version": request.version}
    else:
        raise HTTPException(status_code=500, detail=message)


@app.get("/api/model/status")
async def model_status():
    """현재 모델 상태"""
    return {
        "loaded": tts_engine is not None,
        "version": current_version,
        "gpt_path": current_gpt_path,
        "sovits_path": current_sovits_path,
        "device": device
    }


@app.get("/api/models")
async def list_models():
    """사용 가능한 모델 목록"""
    gpt_models, sovits_models = get_available_models()
    return {
        "versions": MODEL_VERSIONS,
        "custom_gpt": gpt_models,
        "custom_sovits": sovits_models
    }


@app.get("/api/references")
async def list_references():
    """참조 오디오 목록"""
    return get_reference_audios()


@app.get("/api/languages")
async def list_languages():
    """지원 언어 목록"""
    return {
        "supported": list(set(LANGUAGES.keys())),
        "mapping": LANGUAGES
    }


@app.post("/api/tts")
async def synthesize_tts(request: TTSRequest):
    """
    TTS 합성 API (스트리밍)
    
    텍스트를 음성으로 변환하여 스트리밍 또는 전체 오디오로 반환합니다.
    
    - streaming=True: 청크별로 오디오 스트리밍 (실시간 재생에 적합)
    - streaming=False: 전체 오디오를 한번에 반환
    """
    api_start = time.time()
    print(f"\n[API +0.00s] 요청 수신: '{request.text[:30]}...'")
    
    # 참조 오디오 파일 확인
    if not os.path.exists(request.ref_audio_path):
        raise HTTPException(status_code=400, detail=f"참조 오디오 파일이 없습니다: {request.ref_audio_path}")
    
    print(f"[API +{time.time()-api_start:.2f}s] 파일 확인 완료")
    
    # 모델 로드 확인
    if tts_engine is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다. /api/model/load를 먼저 호출하세요.")
    
    print(f"[API +{time.time()-api_start:.2f}s] TTS 시작")
    
    # 스트리밍 모드
    if request.streaming:
        def audio_generator():
            try:
                for chunk in generate_audio_stream(request):
                    yield chunk
            except Exception as e:
                print(f"스트리밍 오류: {e}")
                raise
        
        return StreamingResponse(
            audio_generator(),
            media_type="application/octet-stream",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "X-Accel-Buffering": "no",
                "X-Content-Type": "audio/pcm",
                "X-Sample-Rate": "32000",
                "X-Channels": "1",
                "X-Bit-Depth": "16"
            }
        )
    
    # 비스트리밍 모드
    else:
        audio_bytes, sample_rate, audio_duration, elapsed_time = generate_full_audio(request)
        media_type = "audio/wav" if request.format == "wav" else "audio/pcm"
        
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=tts_output.{request.format}",
                "Content-Length": str(len(audio_bytes)),
                "X-Audio-Duration": str(audio_duration),
                "X-Elapsed-Time": str(elapsed_time),
                "X-RTF": str(elapsed_time / audio_duration) if audio_duration > 0 else "0",
                "X-Sample-Rate": str(sample_rate),
                "X-Channels": "1",
                "X-Bit-Depth": "16"
            }
        )


@app.get("/api/tts/simple")
async def synthesize_simple(
    text: str = Query(..., description="합성할 텍스트"),
    ref_audio: str = Query(..., description="참조 오디오 경로"),
    lang: str = Query("ko", description="언어"),
    ref_text: str = Query("", description="참조 텍스트"),
    speed: float = Query(1.0, ge=0.5, le=2.0, description="속도"),
    streaming: bool = Query(True, description="스트리밍 모드"),
):
    """
    간단한 GET 방식 TTS API
    
    쿼리 파라미터로 간단하게 음성 합성을 요청합니다.
    """
    request = TTSRequest(
        text=text,
        text_lang=lang,
        ref_audio_path=ref_audio,
        ref_text=ref_text,
        ref_lang=lang,
        speed=speed,
        streaming=streaming,
    )
    
    return await synthesize_tts(request)


# ============================================================
# 기본 참조 오디오 설정
# ============================================================

class ReferenceConfig(BaseModel):
    """기본 참조 오디오 설정"""
    audio_path: str = Field(..., description="참조 오디오 파일 경로")
    text: str = Field("", description="참조 오디오의 텍스트")
    lang: str = Field("ko", description="언어 (ko, en, ja, zh)")


@app.post("/api/reference/set")
async def set_default_reference(config: ReferenceConfig):
    """
    기본 참조 오디오 설정
    
    서버에서 참조 오디오를 미리 설정해두면 클라이언트는 텍스트만 전송하면 됩니다.
    참조 오디오 특징을 미리 추출하여 캐시에 저장합니다.
    """
    global DEFAULT_REFERENCE
    
    if not os.path.exists(config.audio_path):
        raise HTTPException(status_code=404, detail=f"파일 없음: {config.audio_path}")
    
    if tts_engine is None:
        raise HTTPException(status_code=503, detail="모델 로드 필요")
    
    DEFAULT_REFERENCE["audio_path"] = config.audio_path
    DEFAULT_REFERENCE["text"] = config.text
    DEFAULT_REFERENCE["lang"] = config.lang
    
    # 참조 오디오 특징 미리 추출 (캐시에 저장)
    print(f"[참조 설정] {os.path.basename(config.audio_path)}")
    print(f"  특징 추출 중...")
    
    import time
    start = time.time()
    try:
        # 참조 오디오 미리 로드 (prompt_semantic, refer_spec 캐싱)
        tts_engine.set_ref_audio(config.audio_path)
        
        # 참조 텍스트도 미리 처리 (phones, bert_features 캐싱)
        if config.text:
            lang_code = LANGUAGES.get(config.lang.lower(), "auto")
            # 텍스트 전처리 호출하여 캐시
            phones, bert_features, norm_text = tts_engine.text_preprocessor.segment_and_extract_feature_for_text(
                config.text, lang_code, tts_engine.configs.version
            )
            tts_engine.prompt_cache["prompt_text"] = config.text
            tts_engine.prompt_cache["prompt_lang"] = lang_code
            tts_engine.prompt_cache["phones"] = phones
            tts_engine.prompt_cache["bert_features"] = bert_features
            tts_engine.prompt_cache["norm_text"] = norm_text
        
        elapsed = time.time() - start
        print(f"  캐싱 완료: {elapsed:.2f}초")
        print(f"  텍스트: {config.text[:50]}..." if len(config.text) > 50 else f"  텍스트: {config.text}")
        
    except Exception as e:
        print(f"  캐싱 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"캐싱 실패: {str(e)}")
    
    return {
        "status": "ok",
        "cached": True,
        "cache_time": elapsed,
        "reference": {
            "audio": os.path.basename(config.audio_path),
            "text": config.text,
            "lang": config.lang
        }
    }


@app.get("/api/reference")
async def get_default_reference():
    """현재 기본 참조 오디오 설정 조회"""
    if not DEFAULT_REFERENCE["audio_path"]:
        return {"status": "not_set"}
    
    return {
        "status": "ok",
        "reference": {
            "audio": os.path.basename(DEFAULT_REFERENCE["audio_path"]),
            "audio_path": DEFAULT_REFERENCE["audio_path"],
            "text": DEFAULT_REFERENCE["text"],
            "lang": DEFAULT_REFERENCE["lang"]
        }
    }


# ============================================================
# 음성 프리셋 관리
# ============================================================

class VoicePreset(BaseModel):
    """음성 프리셋"""
    id: str = Field(..., description="프리셋 ID (영문, 숫자, 밑줄)")
    name: str = Field(..., description="표시 이름")
    audio: str = Field(..., description="참조 오디오 파일 경로")
    text: str = Field("", description="참조 텍스트")
    lang: str = Field("ko", description="언어")


@app.get("/api/presets")
async def list_presets():
    """등록된 음성 프리셋 목록"""
    presets = []
    for pid, preset in VOICE_PRESETS.items():
        presets.append({
            "id": pid,
            "name": preset.get("name", pid),
            "audio": os.path.basename(preset.get("audio", "")),
            "lang": preset.get("lang", "ko"),
            "has_text": bool(preset.get("text", ""))
        })
    
    return {
        "presets": presets,
        "current": CURRENT_PRESET_ID,
        "count": len(presets)
    }


@app.post("/api/presets/add")
async def add_preset(preset: VoicePreset):
    """음성 프리셋 추가"""
    global VOICE_PRESETS
    
    # 파일 존재 확인 (상대 경로 지원)
    audio_path = resolve_audio_path(preset.audio)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail=f"파일 없음: {audio_path}")
    
    # 상대 경로로 저장 (이동성 향상)
    VOICE_PRESETS[preset.id] = {
        "name": preset.name,
        "audio": preset.audio,  # 원본 경로 유지
        "text": preset.text,
        "lang": preset.lang
    }
    
    # 파일로 저장
    save_presets()
    
    print(f"[프리셋 추가] {preset.id}: {preset.name}")
    
    return {"status": "ok", "id": preset.id, "total": len(VOICE_PRESETS)}


@app.delete("/api/presets/{voice_id}")
async def delete_preset(voice_id: str):
    """음성 프리셋 삭제"""
    global VOICE_PRESETS, CURRENT_PRESET_ID
    
    if voice_id not in VOICE_PRESETS:
        raise HTTPException(status_code=404, detail=f"프리셋 없음: {voice_id}")
    
    del VOICE_PRESETS[voice_id]
    
    if CURRENT_PRESET_ID == voice_id:
        CURRENT_PRESET_ID = None
    
    # 파일로 저장
    save_presets()
    
    return {"status": "ok", "deleted": voice_id}


@app.post("/api/presets/select/{voice_id}")
async def select_preset(voice_id: str, lang: str = "ko"):
    """음성 프리셋 선택 및 캐싱
    
    - voice_id: 음성 ID
    - lang: 언어 (다국어 프리셋에서 해당 언어 선택, 없으면 ko로 폴백)
    """
    global CURRENT_PRESET_ID, DEFAULT_REFERENCE
    
    if voice_id not in VOICE_PRESETS:
        raise HTTPException(status_code=404, detail=f"프리셋 없음: {voice_id}")
    
    if tts_engine is None:
        raise HTTPException(status_code=503, detail="모델 로드 필요")
    
    preset = VOICE_PRESETS[voice_id]
    audio_path, ref_text, ref_lang = get_preset_for_lang(preset, lang)
    
    if not audio_path:
        raise HTTPException(status_code=404, detail=f"프리셋에 오디오 없음: {voice_id}")
    
    print(f"[프리셋 선택] {voice_id}: {preset.get('name', '')} ({ref_lang})")
    print(f"  캐싱 중...")
    
    import time
    start = time.time()
    
    try:
        # 참조 오디오 캐싱
        tts_engine.set_ref_audio(audio_path)
        
        # 참조 텍스트 캐싱
        if ref_text:
            lang_code = LANGUAGES.get(ref_lang.lower(), "auto")
            phones, bert_features, norm_text = tts_engine.text_preprocessor.segment_and_extract_feature_for_text(
                ref_text, lang_code, tts_engine.configs.version
            )
            tts_engine.prompt_cache["prompt_text"] = ref_text
            tts_engine.prompt_cache["prompt_lang"] = lang_code
            tts_engine.prompt_cache["phones"] = phones
            tts_engine.prompt_cache["bert_features"] = bert_features
            tts_engine.prompt_cache["norm_text"] = norm_text
        
        # 현재 프리셋 설정
        CURRENT_PRESET_ID = voice_id
        DEFAULT_REFERENCE["audio_path"] = audio_path
        DEFAULT_REFERENCE["text"] = ref_text
        DEFAULT_REFERENCE["lang"] = ref_lang
        
        # 마지막 선택 저장
        save_presets()
        
        elapsed = time.time() - start
        print(f"  완료: {elapsed:.2f}초")
        
    except Exception as e:
        print(f"  실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "status": "ok",
        "voiceId": voice_id,
        "name": preset["name"],
        "cached": True,
        "cache_time": elapsed
    }


@app.get("/api/presets/current")
async def get_current_preset():
    """현재 선택된 프리셋"""
    if not CURRENT_PRESET_ID:
        return {"status": "none"}
    
    preset = VOICE_PRESETS.get(CURRENT_PRESET_ID, {})
    return {
        "status": "ok",
        "id": CURRENT_PRESET_ID,
        "name": preset.get("name", ""),
        "audio": os.path.basename(preset.get("audio", "")),
        "lang": preset.get("lang", "ko")
    }


# ============================================================
# WebSocket 실시간 TTS
# ============================================================

@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    """
    WebSocket 실시간 TTS
    
    연결 후 JSON으로 요청을 보내면 PCM 오디오 바이너리를 스트리밍 반환합니다.
    HTTP보다 빠른 실시간 응답이 가능합니다.
    
    요청 형식:
    {
        "text": "합성할 텍스트",
        "voiceId": "음성ID",  // 또는 ref_audio 직접 지정
        "ref_audio": "참조 오디오 경로",
        "lang": "ko"
    }
    """
    client_id = id(websocket)
    
    # 연결 수락
    await websocket.accept()
    
    # WebSocket API 키 인증 (첫 메시지로 {"api_key": "xxx"} 전송)
    if AUTH_ENABLED and len(api_key_manager.api_keys) > 0:
        try:
            auth_data = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
            auth_message = json.loads(auth_data)
            api_key = auth_message.get("api_key", "")
            
            if not api_key_manager.validate_key(api_key):
                logger.warning(f"[{client_id}] 인증 실패")
                await websocket.send_json({
                    "type": "error",
                    "code": "AUTH_FAILED",
                    "message": "유효하지 않은 API 키입니다."
                })
                await websocket.close(code=4003)
                return
            
            logger.info(f"[{client_id}] 인증 성공 (키: {api_key[:12]}...)")
            await websocket.send_json({
                "type": "auth",
                "status": "success",
                "message": "인증 성공"
            })
            
        except asyncio.TimeoutError:
            logger.warning(f"[{client_id}] 인증 타임아웃 (10초)")
            await websocket.send_json({
                "type": "error",
                "code": "AUTH_TIMEOUT",
                "message": "인증 타임아웃. 연결 후 10초 이내에 {\"api_key\": \"YOUR_KEY\"} 를 전송해주세요."
            })
            await websocket.close(code=4003)
            return
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"[{client_id}] 인증 메시지 오류: {e}")
            await websocket.send_json({
                "type": "error",
                "code": "AUTH_ERROR",
                "message": "인증 메시지 형식 오류. {\"api_key\": \"YOUR_KEY\"} 형식으로 전송해주세요."
            })
            await websocket.close(code=4003)
            return
    
    # 연결 수 제한 체크
    if not await connection_manager.connect(websocket, client_id):
        stats = connection_manager.get_stats()
        logger.warning(f"연결 거부 (최대 {MAX_CONNECTIONS}개 초과): {websocket.client}")
        await websocket.send_text(json.dumps({
            "error": f"서버가 혼잡합니다. ({stats['active_connections']}/{MAX_CONNECTIONS})"
        }))
        await websocket.close(code=1013)
        return
    
    stats = connection_manager.get_stats()
    logger.info(f"클라이언트 연결: {client_id} ({stats['active_connections']}/{MAX_CONNECTIONS})")
    
    try:
        while True:
            # JSON 요청 수신
            data = await websocket.receive_text()
            req = json.loads(data)
            
            text = req.get("text", "")
            voice_id = req.get("voiceId", "")
            ref_audio = req.get("ref_audio", "")
            ref_text = req.get("ref_text", "")
            lang = req.get("lang", "")
            
            # 1. voiceId로 참조 선택
            if voice_id and voice_id in VOICE_PRESETS:
                preset = VOICE_PRESETS[voice_id]
                
                # auto면 텍스트 언어를 먼저 감지
                if not lang or lang.lower() == "auto":
                    detected_lang = detect_language(text)
                    # 감지된 언어로 참조 오디오 선택
                    ref_audio, ref_text, ref_lang = get_preset_for_lang(preset, detected_lang)
                    lang = detected_lang  # TTS도 감지된 언어로
                else:
                    # 지정된 언어로 참조 오디오 선택
                    ref_audio, ref_text, ref_lang = get_preset_for_lang(preset, lang)
            # 2. 기본 참조 사용 (클라이언트가 제공하지 않으면)
            elif not ref_audio and DEFAULT_REFERENCE["audio_path"]:
                ref_audio = DEFAULT_REFERENCE["audio_path"]
                if not ref_text:
                    ref_text = DEFAULT_REFERENCE["text"]
                if not lang:
                    lang = DEFAULT_REFERENCE["lang"]
            
            if not lang:
                lang = "auto"
            
            # 캐시 키 결정 (voiceId가 있으면 사용, 없으면 ref_audio 경로)
            cache_key = voice_id if voice_id else os.path.basename(ref_audio)
            
            # 캐시 상태 확인 (프리셋별 캐시 매니저 사용)
            using_cached = preset_cache_manager.is_current(cache_key)
            has_saved_cache = preset_cache_manager.has_cache(cache_key)
            
            if not text:
                await websocket.send_text(json.dumps({"error": "text 필수"}))
                continue
            
            if not ref_audio:
                await websocket.send_text(json.dumps({"error": "ref_audio 필수 (또는 /api/reference/set으로 기본값 설정)"}))
                continue
            
            if not os.path.exists(ref_audio):
                await websocket.send_text(json.dumps({"error": f"파일 없음: {ref_audio}"}))
                continue
            
            if tts_engine is None:
                await websocket.send_text(json.dumps({"error": "모델 로드 필요"}))
                continue
            
            # TTS 시작 알림
            start_time = time.time()
            
            # 캐시 상태 문자열
            if using_cached:
                cache_status = "현재"
            elif has_saved_cache:
                cache_status = "복원"
            else:
                cache_status = "새로"
            
            # 대기 상태 표시
            stats = connection_manager.get_stats()
            if stats["active_tts"] >= MAX_CONCURRENT_TTS:
                await websocket.send_text(json.dumps({"status": "queued", "position": stats["active_tts"]}))
                logger.info(f"[{client_id}] 대기열 진입 (현재 {stats['active_tts']}개 처리중)")
            
            try:
                # 세마포어로 동시 TTS 제한
                async with tts_semaphore:
                    await connection_manager.update_activity(client_id)
                    await connection_manager.increment_tts(client_id)
                    
                    logger.info(f"[{client_id}] [{cache_status}] 프리셋:{cache_key} TTS: '{text[:25]}...'")
                    await websocket.send_text(json.dumps({"status": "generating"}))
                    
                    # TTS 입력 준비
                    lang_code = LANGUAGES.get(lang.lower(), "auto")
                    inputs = {
                        "text": text,
                        "text_lang": lang_code,
                        "ref_audio_path": ref_audio,
                        "prompt_text": ref_text,
                        "prompt_lang": lang_code,
                        "top_k": TTS_TOP_K,
                        "top_p": TTS_TOP_P,
                        "temperature": TTS_TEMPERATURE,
                        "speed_factor": TTS_SPEED,
                        "text_split_method": TTS_CUT_METHOD,
                        "batch_size": TTS_BATCH_SIZE,
                        "return_fragment": False,
                        "parallel_infer": True,
                        "split_bucket": True,
                        "sample_steps": 8,
                        "seed": -1,
                        "repetition_penalty": 1.35,
                    }
                    
                    # TTS 실행 (스레드풀에서 비동기 실행)
                    def run_tts_sync():
                        # 프리셋 캐시 복원 (현재 프리셋이 아닌 경우)
                        if not preset_cache_manager.is_current(cache_key):
                            if preset_cache_manager.has_cache(cache_key):
                                preset_cache_manager.restore_cache(cache_key)
                                logger.info(f"[캐시] 프리셋 '{cache_key}' 복원됨")
                        
                        all_audio = []
                        sample_rate = 32000
                        for sr, audio in tts_engine.run(inputs):
                            sample_rate = sr
                            all_audio.append(audio)
                        
                        # TTS 완료 후 현재 캐시 저장
                        preset_cache_manager.save_current_cache(cache_key)
                        
                        return all_audio, sample_rate
                    
                    loop = asyncio.get_event_loop()
                    all_audio, sample_rate = await loop.run_in_executor(tts_executor, run_tts_sync)
                    
                    if all_audio:
                        final_audio = np.concatenate(all_audio) if len(all_audio) > 1 else all_audio[0]
                        
                        # 앞부분 잘림 방지: 50ms 무음 추가
                        silence_samples = int(sample_rate * 0.05)
                        silence = np.zeros(silence_samples, dtype=final_audio.dtype)
                        final_audio_with_silence = np.concatenate([silence, final_audio])
                        
                        pcm_bytes = numpy_to_pcm_bytes(final_audio_with_silence)
                        
                        elapsed = time.time() - start_time
                        audio_duration = len(final_audio) / sample_rate
                        
                        # 메타데이터 전송
                        await websocket.send_text(json.dumps({
                            "status": "ready",
                            "sample_rate": sample_rate,
                            "duration": audio_duration,
                            "elapsed": elapsed,
                            "size": len(pcm_bytes)
                        }))
                        
                        # PCM 오디오 바이너리 전송
                        await websocket.send_bytes(pcm_bytes)
                        
                        logger.info(f"[{client_id}] 완료: {elapsed:.2f}초, 오디오 {audio_duration:.2f}초")
                    else:
                        await websocket.send_text(json.dumps({"error": "합성 실패"}))
            
            except Exception as tts_error:
                logger.error(f"[{client_id}] TTS 오류: {tts_error}")
                try:
                    await websocket.send_text(json.dumps({"error": f"TTS 오류: {str(tts_error)}"}))
                except:
                    pass
    
    except WebSocketDisconnect:
        logger.info(f"[{client_id}] 연결 해제")
    except Exception as e:
        logger.error(f"[{client_id}] 연결 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 연결 관리자에서 제거
        await connection_manager.disconnect(client_id)
        stats = connection_manager.get_stats()
        logger.info(f"연결 종료 (남은 연결: {stats['active_connections']}/{MAX_CONNECTIONS})")
        try:
            await websocket.close()
        except:
            pass


# ============================================================
# 빠른 동기 TTS (최소 오버헤드)
# ============================================================

def fast_tts_sync(text: str, ref_audio: str, lang: str = "ko") -> tuple:
    """동기 TTS - 최소 오버헤드"""
    if tts_engine is None:
        return None, 0, "모델 로드 필요"
    
    lang_code = LANGUAGES.get(lang.lower(), "auto")
    inputs = {
        "text": text,
        "text_lang": lang_code,
        "ref_audio_path": ref_audio,
        "prompt_text": "",
        "prompt_lang": lang_code,
        "top_k": TTS_TOP_K,
        "top_p": TTS_TOP_P,
        "temperature": TTS_TEMPERATURE,
        "speed_factor": TTS_SPEED,
        "text_split_method": TTS_CUT_METHOD,
        "batch_size": TTS_BATCH_SIZE,
        "return_fragment": False,
        "parallel_infer": True,
        "split_bucket": True,
        "sample_steps": 8,
        "seed": -1,
    }
    
    all_audio = []
    sample_rate = 32000
    
    for sr, audio in tts_engine.run(inputs):
        sample_rate = sr
        all_audio.append(audio)
    
    if all_audio:
        final_audio = np.concatenate(all_audio) if len(all_audio) > 1 else all_audio[0]
        return final_audio, sample_rate, None
    
    return None, 0, "합성 실패"


# ============================================================
# 메인 실행
# ============================================================

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="GPT-SoVITS TTS API Server")
    parser.add_argument("--mode", choices=["ws", "wss", "both"], default="ws",
                        help="실행 모드: ws(HTTP만), wss(HTTPS만), both(둘 다) - 기본값: ws")
    parser.add_argument("--port", type=int, default=9010, help="WS 포트 (기본: 9010)")
    parser.add_argument("--wss-port", type=int, default=9010, help="WSS 포트 (기본: 9010)")
    # SSL 인증서 기본 경로 (STT 서버와 공유)
    default_ssl_key = "/home/ubuntu/Keys/key.pem"
    default_ssl_cert = "/home/ubuntu/Keys/cert.pem"
    parser.add_argument("--ssl-key", default=default_ssl_key, help="SSL 개인키 경로")
    parser.add_argument("--ssl-cert", default=default_ssl_cert, help="SSL 인증서 경로")
    parser.add_argument("--version", type=str, default=None, 
                        help="모델 버전 (v1, v2, v3, v4, v2Pro)")
    parser.add_argument("--gpt", type=str, default="", help="커스텀 GPT 모델 경로")
    parser.add_argument("--sovits", type=str, default="", help="커스텀 SoVITS 모델 경로")
    parser.add_argument("--no-autoload", action="store_true", help="모델 자동 로드 비활성화")
    parser.add_argument("--generate-key", type=str, metavar="NAME",
                        help="API 키 생성 후 종료 (예: --generate-key myapp)")
    parser.add_argument("--list-keys", action="store_true", help="등록된 API 키 목록 출력")
    parser.add_argument("--no-auth", action="store_true",
                        help="API 키 인증 비활성화 (개발/테스트용)")
    
    args = parser.parse_args()
    
    # 인증 설정
    if args.no_auth:
        AUTH_ENABLED = False
        logger.warning("API 키 인증이 비활성화되었습니다! (--no-auth)")
    
    # API 키 생성 모드
    if args.generate_key:
        key = api_key_manager.generate_key(args.generate_key)
        print(f"\n  API 키 생성 완료!")
        print(f"  Name: {args.generate_key}")
        print(f"  Key:  {key}")
        print(f"\n  사용법:")
        print(f"    REST:      X-API-Key: {key}")
        print(f"    WebSocket: 연결 후 첫 메시지로 {{\"api_key\": \"{key}\"}} 전송")
        print()
        exit(0)
    
    # API 키 목록
    if args.list_keys:
        keys = api_key_manager.list_keys()
        if not keys:
            print("\n  등록된 API 키 없음 (인증 비활성화 상태)")
        else:
            print(f"\n  등록된 API 키 ({len(keys)}개):")
            for info in keys:
                last = info['last_used'] or '사용 안함'
                print(f"    - {info['name']} ({info['key_preview']}, 요청 {info['request_count']}회, 마지막: {last})")
        print()
        exit(0)
    
    PORT = args.port
    WSS_PORT = args.wss_port
    SSL_KEYFILE = args.ssl_key
    SSL_CERTFILE = args.ssl_cert
    
    # API 키가 하나도 없고 인증 활성화 상태면 자동 생성
    if AUTH_ENABLED and len(api_key_manager.api_keys) == 0:
        key = api_key_manager.generate_key("default")
        logger.info(f"기본 API 키 생성됨: {key}")
        logger.info(f"   이 키를 클라이언트에 제공하세요")
    
    # SSL 인증서 존재 여부 확인
    ssl_available = os.path.exists(SSL_KEYFILE) and os.path.exists(SSL_CERTFILE)
    
    # 모델 선택
    if args.no_autoload:
        STARTUP_MODEL["version"] = None
    elif args.version:
        STARTUP_MODEL["version"] = args.version
        STARTUP_MODEL["custom_gpt"] = args.gpt
        STARTUP_MODEL["custom_sovits"] = args.sovits
    else:
        # 대화형 선택
        print("\n" + "=" * 50)
        print("  GPT-SoVITS TTS API Server - 모델 선택")
        print("=" * 50)
        print("\n사용 가능한 모델:")
        print("  [1] v1     - 기본")
        print("  [2] v2     - 개선된 품질")
        print("  [3] v3     - 고품질 (느림)")
        print("  [4] v4     - 최신")
        print("  [5] v2Pro  - 프로페셔널 (권장)")
        print("  [0] 로드 안함 (나중에 API로 로드)")
        
        # 커스텀 모델 확인
        gpt_models, sovits_models = get_available_models()
        if len(gpt_models) > 0 or len(sovits_models) > 0:
            print(f"\n  커스텀 GPT 모델: {len(gpt_models)}개")
            print(f"  커스텀 SoVITS 모델: {len(sovits_models)}개")
        
        choice = input("\n선택 (기본 5-v2Pro): ").strip()
        
        version_map = {
            "1": "v1", "2": "v2", "3": "v3", "4": "v4", "5": "v2Pro",
            "": "v2Pro", "0": None
        }
        
        selected_version = version_map.get(choice, "v2Pro")
        
        if selected_version:
            # 커스텀 모델 선택
            use_custom_gpt = ""
            use_custom_sovits = ""
            
            if gpt_models:
                print(f"\n커스텀 GPT 모델:")
                for i, m in enumerate(gpt_models[:5]):
                    print(f"  [{i+1}] {os.path.basename(m)}")
                gpt_choice = input("GPT 선택 (엔터=기본): ").strip()
                if gpt_choice.isdigit() and 0 < int(gpt_choice) <= len(gpt_models):
                    use_custom_gpt = gpt_models[int(gpt_choice)-1]
            
            if sovits_models:
                print(f"\n커스텀 SoVITS 모델:")
                for i, m in enumerate(sovits_models[:5]):
                    print(f"  [{i+1}] {os.path.basename(m)}")
                sovits_choice = input("SoVITS 선택 (엔터=기본): ").strip()
                if sovits_choice.isdigit() and 0 < int(sovits_choice) <= len(sovits_models):
                    use_custom_sovits = sovits_models[int(sovits_choice)-1]
            
            STARTUP_MODEL["version"] = selected_version
            STARTUP_MODEL["custom_gpt"] = use_custom_gpt
            STARTUP_MODEL["custom_sovits"] = use_custom_sovits
        else:
            STARTUP_MODEL["version"] = None
    
    # 서버 실행 함수
    def run_ws_server():
        """WS (HTTP) 서버 실행"""
        logger.info(f"🌐 WS 서버 시작: ws://0.0.0.0:{PORT}")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=PORT,
            log_level="info",
            ws_ping_interval=30,
            ws_ping_timeout=30
        )
    
    def run_wss_server():
        """WSS (HTTPS) 서버 실행"""
        logger.info(f"🔒 WSS 서버 시작: wss://0.0.0.0:{WSS_PORT}")
        logger.info(f"   - 인증서: {os.path.abspath(SSL_CERTFILE)}")
        logger.info(f"   - 개인키: {os.path.abspath(SSL_KEYFILE)}")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=WSS_PORT,
            log_level="info",
            ws_ping_interval=30,
            ws_ping_timeout=30,
            ssl_keyfile=SSL_KEYFILE,
            ssl_certfile=SSL_CERTFILE
        )
    
    # 시작 메시지 출력
    print(f"\n{'='*60}")
    print(f"  Starting GPT-SoVITS TTS API Server")
    print(f"  Mode: {args.mode.upper()}")
    if args.mode in ["ws", "both"]:
        print(f"  WS:  http://localhost:{PORT}")
        print(f"       ws://localhost:{PORT}/ws/tts")
    if args.mode in ["wss", "both"]:
        print(f"  WSS: https://localhost:{WSS_PORT}")
        print(f"       wss://localhost:{WSS_PORT}/ws/tts")
    print(f"  Docs: http://localhost:{PORT}/docs")
    print(f"  Auth: {'ON' if AUTH_ENABLED else 'OFF'} ({len(api_key_manager.api_keys)}개 키)")
    if STARTUP_MODEL["version"]:
        print(f"  Model: {STARTUP_MODEL['version']}")
        if STARTUP_MODEL["custom_gpt"]:
            print(f"  GPT: {os.path.basename(STARTUP_MODEL['custom_gpt'])}")
        if STARTUP_MODEL["custom_sovits"]:
            print(f"  SoVITS: {os.path.basename(STARTUP_MODEL['custom_sovits'])}")
    else:
        print(f"  Model: (자동 로드 안함)")
    print(f"{'='*60}\n")
    
    # 서버 실행 (모드에 따라)
    if args.mode == "ws":
        # WS만 실행
        run_ws_server()
        
    elif args.mode == "wss":
        # WSS만 실행
        if not ssl_available:
            logger.error("❌ SSL 인증서를 찾을 수 없습니다!")
            logger.error(f"   - 필요: {SSL_KEYFILE}, {SSL_CERTFILE}")
            logger.error("   → --ssl-key, --ssl-cert 경로를 지정하거나 --mode ws 로 시작하세요")
            exit(1)
        run_wss_server()
        
    else:  # both
        # WS와 WSS 둘 다 실행
        if not ssl_available:
            logger.warning("⚠️ SSL 인증서가 없어서 WS 모드로만 실행합니다")
            logger.warning(f"   → WSS도 사용하려면 {SSL_KEYFILE}, {SSL_CERTFILE} 파일을 생성하세요")
            run_ws_server()
        else:
            logger.info("🌐 WS + WSS 동시 실행 모드")
            
            # WSS를 별도 스레드에서 실행
            wss_thread = threading.Thread(target=run_wss_server, daemon=True)
            wss_thread.start()
            
            # WS는 메인 스레드에서 실행
            run_ws_server()
