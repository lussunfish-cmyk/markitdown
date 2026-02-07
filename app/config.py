"""
Ollama 기반 RAG 시스템의 설정.
"""

import os
from pathlib import Path


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
    
    # 분할 구분자
    SEPARATORS = ["\n\n", "\n", ".", " ", ""]


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


class AppConfig:
    """애플리케이션 설정."""
    
    # 로깅
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # API 설정
    API_TITLE = "MarkItDown RAG API"
    API_VERSION = "1.0.0"
    
    # 하위 설정
    OLLAMA = OllamaConfig()
    VECTOR_STORE = VectorStoreConfig()
    CHUNKING = ChunkingConfig()
    RAG = RAGConfig()
    CONVERSION = ConversionConfig()
    INDEXING = IndexingConfig()
    BATCH = BatchConfig()


# 전역 설정 인스턴스
config = AppConfig()
