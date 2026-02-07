# config.py 구현

## 개요

Ollama 기반 RAG 시스템의 전체 설정을 관리하는 모듈입니다. 환경 변수를 통해 설정을 로드하고, 각 기능별로 구조화된 설정 클래스를 제공합니다.

## 파일 경로

```
markitdown/app/config.py
```

## 주요 구성 요소

### 1. OllamaConfig

Ollama 서버 연결 및 모델 설정을 관리합니다.

```python
class OllamaConfig:
    """Ollama 서버 설정."""
    
    # Ollama 서버 설정
    BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # 모델 설정
    EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "gemma2")
    
    # 타임아웃 및 재시도 설정
    REQUEST_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))
    MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
    RETRY_DELAY = float(os.getenv("OLLAMA_RETRY_DELAY", "1.0"))
```

**주요 속성:**
- `BASE_URL`: Ollama 서버 주소 (기본값: `http://localhost:11434`)
- `EMBEDDING_MODEL`: 임베딩 모델명 (기본값: `nomic-embed-text`)
- `LLM_MODEL`: 텍스트 생성 모델명 (기본값: `gemma2`)
- `REQUEST_TIMEOUT`: API 요청 타임아웃 (초 단위, 기본값: 300초)
- `MAX_RETRIES`: 최대 재시도 횟수 (기본값: 3회)
- `RETRY_DELAY`: 재시도 간격 (초 단위, 기본값: 1.0초)

### 2. VectorStoreConfig

벡터 저장소 설정을 관리합니다.

```python
class VectorStoreConfig:
    """벡터 저장소 설정."""
    
    # 벡터 저장소 타입: "chroma" 또는 "faiss"
    STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")
    
    # 저장소 경로
    PERSIST_DIR = Path(os.getenv("VECTOR_STORE_DIR", "/app/vector_store"))
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Chroma 특정 설정
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "documents")
    
    # 벡터 차원 (nomic-embed-text 기준)
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
```

**주요 속성:**
- `STORE_TYPE`: 벡터 DB 타입 (`chroma` 또는 `faiss`)
- `PERSIST_DIR`: 벡터 데이터 저장 경로
- `CHROMA_COLLECTION_NAME`: ChromaDB 컬렉션명
- `EMBEDDING_DIM`: 임베딩 벡터 차원 (nomic-embed-text는 768차원)

### 3. ChunkingConfig

텍스트 분할(청킹) 설정을 관리합니다.

```python
class ChunkingConfig:
    """텍스트 분할 설정."""
    
    # 청크 크기 (문자 단위)
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
    
    # 청크 간 겹침 (문자 단위)
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "256"))
    
    # 분할 구분자
    SEPARATORS = ["\n\n", "\n", ".", " ", ""]
```

**주요 속성:**
- `CHUNK_SIZE`: 각 청크의 최대 크기 (문자 단위, 기본값: 1024)
- `CHUNK_OVERLAP`: 청크 간 중복 영역 크기 (기본값: 256)
- `SEPARATORS`: 텍스트 분할 시 사용할 구분자 우선순위

### 4. RAGConfig

RAG (검색 증강 생성) 동작 설정을 관리합니다.

```python
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
    MAX_CONTEXT_LENGTH = int(os.getenv("RAG_MAX_CONTEXT_LENGTH", "4096"))
```

**주요 속성:**
- `TOP_K`: 검색할 상위 문서 개수 (기본값: 3)
- `SIMILARITY_THRESHOLD`: 최소 유사도 임계값 (0~1, 기본값: 0.5)
- `TEMPERATURE`: LLM 생성 온도 (낮을수록 결정적, 기본값: 0.7)
- `TOP_P`: Nucleus sampling 파라미터 (기본값: 0.9)
- `MAX_TOKENS`: 최대 생성 토큰 수 (기본값: 512)
- `MAX_CONTEXT_LENGTH`: 최대 컨텍스트 길이 (기본값: 4096)

### 5. ConversionConfig

파일 변환 설정을 관리합니다.

```python
class ConversionConfig:
    """파일 변환 설정."""
    
    # 입출력 디렉토리
    OUTPUT_DIR = Path(os.getenv("MARKITDOWN_OUTPUT_DIR", "/app/output"))
    INPUT_DIR = Path(os.getenv("MARKITDOWN_INPUT_DIR", "/app/input"))
    
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
```

**주요 속성:**
- `OUTPUT_DIR`: 변환된 파일 저장 경로
- `INPUT_DIR`: 입력 파일 경로
- `SUPPORTED_FORMATS`: 지원하는 파일 확장자 집합
- `LIBREOFFICE_TIMEOUT`: LibreOffice 변환 타임아웃

### 6. IndexingConfig

문서 인덱싱 설정을 관리합니다.

```python
class IndexingConfig:
    """문서 인덱싱 설정."""
    
    # 입출력 디렉토리
    DOCUMENT_DIR = Path(os.getenv("DOCUMENT_DIR", "/app/output"))
    INDEX_STATE_FILE = Path(os.getenv("INDEX_STATE_FILE", "/app/vector_store/index_state.json"))
    
    # 지원 파일 형식
    SUPPORTED_FORMATS = {".md", ".txt"}
    
    # 감시 모드 설정
    WATCH_INTERVAL = int(os.getenv("WATCH_INTERVAL", "10"))
```

**주요 속성:**
- `DOCUMENT_DIR`: 인덱싱할 문서 디렉토리
- `INDEX_STATE_FILE`: 인덱싱 상태 저장 파일
- `SUPPORTED_FORMATS`: 인덱싱 지원 파일 형식
- `WATCH_INTERVAL`: 파일 감시 간격 (초)

### 7. AppConfig

전체 애플리케이션 설정을 통합 관리합니다.

```python
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

# 전역 설정 인스턴스
config = AppConfig()
```

## 사용 예시

### 설정 임포트 및 사용

```python
from config import config

# Ollama 서버 URL 접근
print(config.OLLAMA.BASE_URL)

# 벡터 저장소 경로 접근
print(config.VECTOR_STORE.PERSIST_DIR)

# RAG 파라미터 접근
top_k = config.RAG.TOP_K
temperature = config.RAG.TEMPERATURE
```

### 환경 변수 설정

Docker Compose에서 환경 변수를 통해 설정을 변경할 수 있습니다:

```yaml
environment:
  - OLLAMA_BASE_URL=http://ollama-server:11434
  - OLLAMA_LLM_MODEL=llama2
  - RAG_TOP_K=5
  - CHUNK_SIZE=2048
  - VECTOR_STORE_DIR=/data/vectors
```

## 환경 변수 전체 목록

| 환경 변수 | 기본값 | 설명 |
|----------|--------|------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 서버 주소 |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | 임베딩 모델 |
| `OLLAMA_LLM_MODEL` | `gemma2` | LLM 모델 |
| `OLLAMA_TIMEOUT` | `300` | API 타임아웃 (초) |
| `OLLAMA_MAX_RETRIES` | `3` | 최대 재시도 횟수 |
| `OLLAMA_RETRY_DELAY` | `1.0` | 재시도 간격 (초) |
| `VECTOR_STORE_TYPE` | `chroma` | 벡터 DB 타입 |
| `VECTOR_STORE_DIR` | `/app/vector_store` | 벡터 저장 경로 |
| `CHROMA_COLLECTION_NAME` | `documents` | Chroma 컬렉션명 |
| `EMBEDDING_DIM` | `768` | 임베딩 차원 |
| `CHUNK_SIZE` | `1024` | 청크 크기 |
| `CHUNK_OVERLAP` | `256` | 청크 겹침 |
| `RAG_TOP_K` | `3` | 검색 결과 개수 |
| `RAG_SIMILARITY_THRESHOLD` | `0.5` | 유사도 임계값 |
| `RAG_TEMPERATURE` | `0.7` | LLM 온도 |
| `RAG_TOP_P` | `0.9` | Top-p 샘플링 |
| `RAG_MAX_TOKENS` | `512` | 최대 토큰 수 |
| `RAG_MAX_CONTEXT_LENGTH` | `4096` | 최대 컨텍스트 길이 |
| `MARKITDOWN_OUTPUT_DIR` | `/app/output` | 변환 출력 경로 |
| `MARKITDOWN_INPUT_DIR` | `/app/input` | 변환 입력 경로 |
| `LIBREOFFICE_TIMEOUT` | `60` | LibreOffice 타임아웃 |
| `DOCUMENT_DIR` | `/app/output` | 문서 디렉토리 |
| `INDEX_STATE_FILE` | `/app/vector_store/index_state.json` | 인덱싱 상태 파일 |
| `WATCH_INTERVAL` | `10` | 파일 감시 간격 (초) |
| `LOG_LEVEL` | `INFO` | 로깅 레벨 |

## 설계 원칙

1. **환경 변수 우선**: 모든 설정은 환경 변수로 오버라이드 가능
2. **합리적인 기본값**: 각 설정에 적절한 기본값 제공
3. **타입 안전성**: 환경 변수를 적절한 타입으로 변환
4. **계층적 구조**: 기능별로 설정을 그룹화하여 관리 용이
5. **전역 싱글톤**: `config` 인스턴스를 통해 어디서든 접근 가능

## 확장 방법

새로운 설정을 추가하려면:

1. 해당 기능의 Config 클래스에 속성 추가
2. 환경 변수를 통해 로드하도록 구현
3. 적절한 기본값 설정
4. 문서에 환경 변수 정보 추가

```python
class NewFeatureConfig:
    """새 기능 설정."""
    
    NEW_PARAM = os.getenv("NEW_PARAM", "default_value")
    
class AppConfig:
    # ...
    NEW_FEATURE = NewFeatureConfig()
```
