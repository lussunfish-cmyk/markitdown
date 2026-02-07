"""
Ollama 기반 RAG 시스템의 설정.

모든 설정값은 환경 변수로 오버라이드 가능하며,
각 클래스는 관련된 설정을 논리적으로 그룹화합니다.
"""

import os
from pathlib import Path
from typing import List


# ============================================================================
# 로깅 설정
# ============================================================================

class LoggingConfig:
    """로깅 관련 설정."""
    
    # 로그 레벨
    LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # 로그 포맷
    FORMAT = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 날짜 포맷
    DATE_FORMAT = os.getenv("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")


# ============================================================================
# Ollama 설정
# ============================================================================

class OllamaConfig:
    """Ollama 서버 설정."""
    
    # Ollama 서버 설정
    BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # 모델 설정
    EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
    LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "gemma2")
    
    # 타임아웃 및 재시도 설정
    REQUEST_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))
    MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
    RETRY_DELAY = float(os.getenv("OLLAMA_RETRY_DELAY", "1.0"))


class VectorStoreConfig:
    """벡터 저장소 설정."""
    
    # 벡터 저장소 타입: "chroma" 또는 "faiss"
    STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")
    
    # 저장소 경로
    PERSIST_DIR = Path(os.getenv("VECTOR_STORE_DIR", "/app/vector_store"))
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Chroma 특정 설정
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "documents")
    
    # 벡터 차원 (multilingual-e5-large 기준)
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))


class ChunkingConfig:
    """텍스트 분할 설정."""
    
    # 청크 크기 (문자 단위) - mxbai-embed-large 토큰 제한 고려
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    
    # 청크 간 겹침 (문자 단위)
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "128"))
    
    # 분할 구분자 (우선순위 순서)
    SEPARATORS = ["\n\n", "\n", ".", " ", ""]


# ============================================================================
# 검색(Retriever) 설정
# ============================================================================

class RetrieverConfig:
    """문서 검색 설정."""
    
    # 하이브리드 검색 가중치 (벡터 검색 vs 키워드 검색)
    # alpha=1.0: 벡터 검색만, alpha=0.0: 키워드 검색만
    HYBRID_ALPHA = float(os.getenv("RETRIEVER_HYBRID_ALPHA", "0.7"))
    
    # RRF (Reciprocal Rank Fusion) 파라미터
    RRF_K = int(os.getenv("RETRIEVER_RRF_K", "60"))
    
    # 기본 검색 타입 ("vector", "bm25", "hybrid", "advanced")
    DEFAULT_TYPE = os.getenv("RETRIEVER_DEFAULT_TYPE", "advanced")


# ============================================================================
# 캐시 설정
# ============================================================================

class CacheConfig:
    """RAG 캐싱 설정."""
    
    # 검색 결과 캐시 활성화
    ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    
    # 검색 결과 캐시 최대 크기
    SEARCH_CACHE_MAXSIZE = int(os.getenv("CACHE_SEARCH_MAXSIZE", "100"))
    
    # 검색 결과 캐시 TTL (초)
    SEARCH_CACHE_TTL = int(os.getenv("CACHE_SEARCH_TTL", "300"))


# ============================================================================
# RAG 설정
# ============================================================================

class RAGConfig:
    """RAG(검색 증강 생성) 설정."""
    
    # 검색 설정
    TOP_K = int(os.getenv("RAG_TOP_K", "3"))
    SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.5"))
    
    # LLM 생성 설정
    TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.7"))
    TOP_P = float(os.getenv("RAG_TOP_P", "0.9"))
    MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", "512"))
    
    # 컨텍스트 설정
    MAX_CONTEXT_LENGTH = int(os.getenv("RAG_MAX_CONTEXT_LENGTH", "8192"))


class ConversionConfig:
    """파일 변환 설정."""
    
    # 입출력 디렉토리
    OUTPUT_DIR = Path(os.getenv("MARKITDOWN_OUTPUT_DIR", "/app/output"))
    INPUT_DIR = Path(os.getenv("MARKITDOWN_INPUT_DIR", "/app/input"))
    
    # 디렉토리 생성
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 지원 파일 형식
    SUPPORTED_FORMATS = {
        '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls',
        '.csv', '.json', '.xml', '.html', '.htm', '.txt', '.md',
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff',
        '.wav', '.mp3', '.m4a', '.flac', '.epub', '.zip'
    }
    
    # LibreOffice 타임아웃 (초)
    LIBREOFFICE_TIMEOUT = int(os.getenv("LIBREOFFICE_TIMEOUT", "60"))
    
    # 결과 파일명
    RESULT_FILENAME = "conversion_result.json"


class IndexingConfig:
    """문서 인덱싱 설정."""
    
    # 입출력 디렉토리
    DOCUMENT_DIR = Path(os.getenv("DOCUMENT_DIR", "/app/output"))
    INDEX_STATE_FILE = Path(os.getenv("INDEX_STATE_FILE", "/app/vector_store/index_state.json"))
    
    # 지원 파일 형식
    SUPPORTED_FORMATS = {".md", ".txt"}
    
    # 감시 모드 설정
    WATCH_INTERVAL = int(os.getenv("WATCH_INTERVAL", "10"))


class BatchConfig:
    """배치 처리 설정."""
    
    # 배치 상태 저장 경로
    BATCH_STATE_DIR = Path(os.getenv("BATCH_STATE_DIR", "/app/batch_state"))
    BATCH_STATE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 기본 배치 크기
    DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "100"))
    
    # 배치 처리 타임아웃 (초, 0이면 무제한)
    BATCH_TIMEOUT = int(os.getenv("BATCH_TIMEOUT", "0"))
    
    # 배치 ID 생성 시 해시 길이
    HASH_SUFFIX_LENGTH = int(os.getenv("BATCH_HASH_LENGTH", "6"))


# ============================================================================
# API 설정
# ============================================================================

class APIConfig:
    """REST API 설정."""
    
    # API 타이틀
    TITLE = os.getenv("API_TITLE", "MarkItDown RAG API")
    
    # API 버전
    VERSION = os.getenv("API_VERSION", "1.0.0")
    
    # API 설명
    DESCRIPTION = os.getenv(
        "API_DESCRIPTION",
        "문서 변환, 벡터 인덱싱, RAG 기반 질의응답 API"
    )
    
    # CORS 설정
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # API 문서 경로
    DOCS_URL = os.getenv("API_DOCS_URL", "/docs")
    REDOC_URL = os.getenv("API_REDOC_URL", "/redoc")


# ============================================================================
# 통합 설정
# ============================================================================

class AppConfig:
    """애플리케이션 통합 설정."""
    
    # 하위 설정 인스턴스
    LOGGING = LoggingConfig()
    OLLAMA = OllamaConfig()
    VECTOR_STORE = VectorStoreConfig()
    CHUNKING = ChunkingConfig()
    RETRIEVER = RetrieverConfig()
    CACHE = CacheConfig()
    RAG = RAGConfig()
    CONVERSION = ConversionConfig()
    INDEXING = IndexingConfig()
    BATCH = BatchConfig()
    API = APIConfig()
    
    @classmethod
    def get_all_settings(cls) -> dict:
        """모든 설정을 딕셔너리로 반환 (디버깅용)."""
        settings = {}
        for attr_name in dir(cls):
            if attr_name.isupper():
                continue
            attr = getattr(cls, attr_name)
            if isinstance(attr, type):
                settings[attr_name] = {
                    k: v for k, v in vars(attr).items() 
                    if not k.startswith('_')
                }
        return settings


# ============================================================================
# 전역 설정 인스턴스
# ============================================================================

# 전역 설정 인스턴스
config = AppConfig()


# ============================================================================
# 설정 검증
# ============================================================================

def validate_config() -> None:
    """설정값 유효성 검증."""
    # 청크 크기 검증
    if config.CHUNKING.CHUNK_OVERLAP >= config.CHUNKING.CHUNK_SIZE:
        raise ValueError(
            f"CHUNK_OVERLAP({config.CHUNKING.CHUNK_OVERLAP})은 "
            f"CHUNK_SIZE({config.CHUNKING.CHUNK_SIZE})보다 작아야 합니다"
        )
    
    # alpha 범위 검증
    if not 0.0 <= config.RETRIEVER.HYBRID_ALPHA <= 1.0:
        raise ValueError(
            f"HYBRID_ALPHA({config.RETRIEVER.HYBRID_ALPHA})는 "
            f"0.0 ~ 1.0 사이여야 합니다"
        )
    
    # temperature 범위 검증
    if not 0.0 <= config.RAG.TEMPERATURE <= 2.0:
        raise ValueError(
            f"TEMPERATURE({config.RAG.TEMPERATURE})는 "
            f"0.0 ~ 2.0 사이여야 합니다"
        )


# 모듈 로드 시 설정 검증
validate_config()
