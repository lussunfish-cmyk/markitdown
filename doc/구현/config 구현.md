# config.py 구현

## 개요

Ollama 기반 RAG 시스템의 전체 설정을 관리하는 모듈입니다. 모든 설정은 환경 변수를 통해 오버라이드할 수 있으며, 관련 설정을 논리적인 클래스로 그룹화하여 제공합니다.

## 파일 경로

```
markitdown/app/config.py
```

## 주요 구성 요소

### 1. LoggingConfig

로깅 관련 설정을 관리합니다.

```python
class LoggingConfig:
    """로깅 관련 설정."""
    LEVEL = os.getenv("LOG_LEVEL", "INFO")
    FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    DATE_FORMAT = os.getenv("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")
```

### 2. OllamaConfig

Ollama 서버 연결 및 모델 설정을 관리합니다.

```python
class OllamaConfig:
    """Ollama 서버 설정."""
    BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
    LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "gemma2")
    REQUEST_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))
    MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
    RETRY_DELAY = float(os.getenv("OLLAMA_RETRY_DELAY", "1.0"))
```

**주요 속성:**
- `EMBEDDING_MODEL`: 임베딩 모델 (기본값: `mxbai-embed-large`)
- `LLM_MODEL`: 생성 모델 (기본값: `gemma2`)

### 3. VectorStoreConfig

벡터 저장소(ChromaDB 등) 설정을 관리합니다.

```python
class VectorStoreConfig:
    """벡터 저장소 설정."""
    STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")
    PERSIST_DIR = Path(os.getenv("VECTOR_STORE_DIR", "/app/vector_store"))
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "documents")
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
```

**주요 속성:**
- `EMBEDDING_DIM`: 벡터 차원 (mxbai-embed-large/multilingual-e5-large 기준 1024)

### 4. ChunkingConfig

텍스트 분할(Chunking) 설정을 관리합니다.

```python
class ChunkingConfig:
    """텍스트 분할 설정."""
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    SEPARATORS = ["\n\n", "\n", ".", " ", ""]
    MD_SEPARATORS = [
        "\n## ", "\n### ", "\n#### ",
        "\n\n", "\n", ". ", " ", ""
    ]
```

**주요 변경:**
- `CHUNK_SIZE`가 800으로 조정되어 LLM 컨텍스트 활용 최적화
- `MD_SEPARATORS` 추가로 마크다운 헤더 기반 분할 지원

### 5. RetrieverConfig

검색 및 리랭킹(Reranking) 관련 설정입니다.

```python
class RetrieverConfig:
    """문서 검색 설정."""
    HYBRID_ALPHA = float(os.getenv("RETRIEVER_HYBRID_ALPHA", "0.8"))
    RRF_K = int(os.getenv("RETRIEVER_RRF_K", "60"))
    DEFAULT_TYPE = os.getenv("RETRIEVER_DEFAULT_TYPE", "advanced")
    RERANKER_MODEL = os.getenv("RETRIEVER_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    USE_RERANKER = os.getenv("RETRIEVER_USE_RERANKER", "true").lower() == "true"
```

**주요 속성:**
- `HYBRID_ALPHA`: 하이브리드 검색 시 벡터 검색 가중치 (0.8)
- `RERANKER_MODEL`: 리랭킹 모델로 `BAAI/bge-reranker-v2-m3` 사용 (다국어 성능 우수)

### 6. CacheConfig

검색 결과 캐싱 설정입니다.

```python
class CacheConfig:
    """RAG 캐싱 설정."""
    ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    SEARCH_CACHE_MAXSIZE = int(os.getenv("CACHE_SEARCH_MAXSIZE", "100"))
    SEARCH_CACHE_TTL = int(os.getenv("CACHE_SEARCH_TTL", "300"))
```

### 7. RAGConfig

RAG 생성 파라미터 및 컨텍스트 설정입니다.

```python
class RAGConfig:
    """RAG(검색 증강 생성) 설정."""
    TOP_K = int(os.getenv("RAG_TOP_K", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.5"))
    TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.3"))
    TOP_P = float(os.getenv("RAG_TOP_P", "0.9"))
    MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", "2048"))
    MAX_CONTEXT_LENGTH = int(os.getenv("RAG_MAX_CONTEXT_LENGTH", "8192"))
    ENABLE_QUERY_REWRITING = os.getenv("RAG_ENABLE_QUERY_REWRITING", "true").lower() == "true"
```

### 8. ConversionConfig

문서 변환 관련 설정입니다.

```python
class ConversionConfig:
    """파일 변환 설정."""
    OUTPUT_DIR = Path(os.getenv("MARKITDOWN_OUTPUT_DIR", "/app/output"))
    INPUT_DIR = Path(os.getenv("MARKITDOWN_INPUT_DIR", "/app/input"))
    SUPPORTED_FORMATS = {...}
    LIBREOFFICE_TIMEOUT = int(os.getenv("LIBREOFFICE_TIMEOUT", "60"))
    RESULT_FILENAME = "conversion_result.json"
```

### 9. IndexingConfig

문서 인덱싱 프로세스 설정입니다.

```python
class IndexingConfig:
    """문서 인덱싱 설정."""
    DOCUMENT_DIR = Path(os.getenv("DOCUMENT_DIR", "/app/output"))
    INDEX_STATE_FILE = Path(os.getenv("INDEX_STATE_FILE", "/app/vector_store/index_state.json"))
    SUPPORTED_FORMATS = {".md", ".txt"}
    WATCH_INTERVAL = int(os.getenv("WATCH_INTERVAL", "10"))
```

### 10. BatchConfig

대량 파일 처리(배치) 설정입니다.

```python
class BatchConfig:
    """배치 처리 설정."""
    BATCH_STATE_DIR = Path(os.getenv("BATCH_STATE_DIR", "/app/batch_state"))
    DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "100"))
    BATCH_TIMEOUT = int(os.getenv("BATCH_TIMEOUT", "0"))
    HASH_SUFFIX_LENGTH = int(os.getenv("BATCH_HASH_LENGTH", "6"))
```

### 11. APIConfig

FastAPI 서버 관련 설정입니다.

```python
class APIConfig:
    """REST API 설정."""
    TITLE = os.getenv("API_TITLE", "MarkItDown RAG API")
    VERSION = os.getenv("API_VERSION", "1.0.0")
    DESCRIPTION = "문서 변환, 벡터 인덱싱, RAG 기반 질의응답 API"
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    DOCS_URL = os.getenv("API_DOCS_URL", "/docs")
    REDOC_URL = os.getenv("API_REDOC_URL", "/redoc")
```

### 12. PromptConfig

시스템 프롬프트 및 사용자 템플릿을 정의합니다. `SYSTEM_PROMPT`는 기술 문서 응답에 최적화된 역할을 부여하며, `QUERY_REWRITE_TEMPLATE`은 검색 쿼리 최적화를 위해 사용됩니다.

## 데이터 흐름

1. OS 환경 변수 (Environment Variables) 로드
2. `config.py`의 각 클래스 (`OllamaConfig`, `RAGConfig` 등) 속성으로 매핑
3. 애플리케이션 시작 시 `config` 객체 참조하여 서비스 초기화
