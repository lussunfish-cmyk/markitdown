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


# ============================================================================
# LM Studio 설정
# ============================================================================

class LMStudioConfig:
    """LM Studio 서버 설정."""
    
    # LM Studio 서버 설정 (LLM 생성용)
    BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234")
    
    # LLM 모델 설정 (LM Studio에서 로드된 모델)
    LLM_MODEL = os.getenv("LMSTUDIO_LLM_MODEL", "qwen3-32b")
    
    # 임베딩 서비스 설정 (별도 서비스로 실행, OpenAI 호환 API)
    # 예: http://localhost:8001 (실제 서비스 URL로 변경)
    EMBEDDING_SERVICE_BASE_URL = os.getenv(
        "LMSTUDIO_EMBEDDING_SERVICE_URL",
        "http://localhost:8001"
    )
    
    # 임베딩 서비스에서 사용할 모델명 (API 요청시 모델 파라미터로 전송)
    EMBEDDING_MODEL = os.getenv(
        "LMSTUDIO_EMBEDDING_MODEL",
        "mxbai-embed-large-v1"
    )
    
    # 타임아웃 및 재시도 설정
    REQUEST_TIMEOUT = int(os.getenv("LMSTUDIO_TIMEOUT", "300"))
    MAX_RETRIES = int(os.getenv("LMSTUDIO_MAX_RETRIES", "3"))
    RETRY_DELAY = float(os.getenv("LMSTUDIO_RETRY_DELAY", "1.0"))


# ============================================================================
# LLM 백엔드 설정
# ============================================================================

class LLMBackendConfig:
    """LLM 백엔드 선택 설정."""
    
    # 사용할 LLM 백엔드 ("ollama" 또는 "lmstudio")
    BACKEND_TYPE = os.getenv("LLM_BACKEND_TYPE", "lmstudio")


class VectorStoreConfig:
    """벡터 저장소 설정."""
    
    # 벡터 저장소 타입: "chroma" 또는 "faiss"
    # FAISS: 대규모 데이터셋에 최적화, 검색 속도 ~1.9배 빠름
    # ChromaDB: 메타데이터 관리 우수, 더 작은 데이터셋에 권장
    STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")
    
    # 저장소 경로
    PERSIST_DIR = Path(os.getenv("VECTOR_STORE_DIR", "./vector_store"))
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Chroma 특정 설정
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "documents")
    
    # FAISS 인덱스 타입 설정
    # - "flat": L2 기반 순차 검색 (메모리 효율, ~40MB/10k docs@1024dim)
    # - "ivf": Inverted File Index (빠른 검색, 메모리 효율, 10k+문서 권장)
    # - "hnsw": 그래프 기반 (가장 빠름, 높은 메모리 사용)
    FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "flat")
    
    # FAISS GPU 사용 여부 (Apple Silicon의 Metal은 현재 공식 미지원)
    FAISS_USE_GPU = os.getenv("FAISS_USE_GPU", "false").lower() == "true"
    
    # 벡터 차원 (multilingual-e5-large 기준)
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))


class ChunkingConfig:
    """텍스트 분할 설정."""
    
    # 청크 크기 (문자 단위) - gemma2를 위해 더 큰 컨텍스트 제공
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    
    # 청크 간 겹침 (문자 단위) - 컨텍스트 연속성 향상
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # 분할 구분자 (우선순위 순서)
    SEPARATORS = ["\n\n", "\n", ".", " ", ""]
    
    # 마크다운 구조를 고려한 구분자
    MD_SEPARATORS = [
        "\n## ", "\n### ", "\n#### ",
        "\n\n", "\n", ". ", " ", ""
    ]


# ============================================================================
# 검색(Retriever) 설정
# ============================================================================

class RetrieverConfig:
    """문서 검색 설정."""
    
    # 하이브리드 검색 가중치 (벡터 검색 vs 키워드 검색)
    # alpha=1.0: 벡터 검색만, alpha=0.0: 키워드 검색만
    # 0.8로 조정: 벡터 검색을 더 선호하면서 키워드도 고려
    HYBRID_ALPHA = float(os.getenv("RETRIEVER_HYBRID_ALPHA", "0.5"))
    
    # RRF (Reciprocal Rank Fusion) 파라미터
    RRF_K = int(os.getenv("RETRIEVER_RRF_K", "60"))
    
    # 기본 검색 타입 ("vector", "bm25", "hybrid", "advanced")
    DEFAULT_TYPE = os.getenv("RETRIEVER_DEFAULT_TYPE", "advanced")
    
    # 리랭커 설정
    # 기존 English 모델인 'cross-encoder/ms-marco-MiniLM-L-6-v2'에서
    # 다국어 지원 및 성능이 월등한 'BAAI/bge-reranker-v2-m3'로 변경
    RERANKER_MODEL = os.getenv("RETRIEVER_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    USE_RERANKER = os.getenv("RETRIEVER_USE_RERANKER", "true").lower() == "true"


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
    
    # 검색 설정 (더 많은 컨텍스트로 품질 향상)
    TOP_K = int(os.getenv("RAG_TOP_K", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.5"))
    
    # LLM 생성 설정 (RAG에 최적화: 낮은 temperature로 더 정확한 답변)
    TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.3"))
    TOP_P = float(os.getenv("RAG_TOP_P", "0.9"))
    MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", "2048"))
    
    # 컨텍스트 설정
    MAX_CONTEXT_LENGTH = int(os.getenv("RAG_MAX_CONTEXT_LENGTH", "8192"))
    
    # 쿼리 최적화 설정
    ENABLE_QUERY_REWRITING = os.getenv("RAG_ENABLE_QUERY_REWRITING", "true").lower() == "true"


class ConversionConfig:
    """파일 변환 설정."""
    
    # 입출력 디렉토리
    OUTPUT_DIR = Path(os.getenv("MARKITDOWN_OUTPUT_DIR", "./output"))
    INPUT_DIR = Path(os.getenv("MARKITDOWN_INPUT_DIR", "./input"))
    
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
    DOCUMENT_DIR = Path(os.getenv("DOCUMENT_DIR", "./output"))
    INDEX_STATE_FILE = Path(os.getenv("INDEX_STATE_FILE", "./vector_store/index_state.json"))
    
    # 지원 파일 형식
    SUPPORTED_FORMATS = {".md", ".txt"}
    
    # 감시 모드 설정
    WATCH_INTERVAL = int(os.getenv("WATCH_INTERVAL", "10"))


class BatchConfig:
    """배치 처리 설정."""
    
    # 배치 상태 저장 경로
    BATCH_STATE_DIR = Path(os.getenv("BATCH_STATE_DIR", "./batch_state"))
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
# 프롬프트 설정 (RAG)
# ============================================================================

class PromptConfig:
    """프롬프트 템플릿 설정."""
    
    # 시스템 프롬프트
    SYSTEM_PROMPT = """You are an expert Technical Writer and Analyst. Your goal is to provide a comprehensive, structured, and easy-to-understand answer in Korean based on the provided English reference documents.

### **CRITICAL GUIDELINES**:
1.  **SOURCE MATERIAL**: Use ONLY the provided provided "Reference Documents". Do not use outside knowledge. If the answer is not in the text, state "제공된 문서에 관련 정보가 없습니다."
2.  **LANGUAGE**: The user asks in Korean, and you MUST answer in **Korean**.
3.  **TONE & STYLE**: 
    - Use a professional, explanatory tone.
    - Do NOT just translate or list facts. **Explain** the concepts so a reader can understand the "Why" and "How".
    - Connect related concepts logically.
4.  **TERMINOLOGY**: 
    - Keep English technical acronyms (e.g., PCEF, gNB, 5G Core, AMF) in **English**.
    - **EXPLAIN** the term in Korean when first used or within the context.
      - Bad: "PCEF는 Qos를 수행합니다."
      - Good: "PCEF(Policy and Charging Enforcement Function)는 정책 및 과금 집행 기능을 담당하며, 사용자의 QoS를 제어합니다 [1]."
5.  **CITATIONS**: precise citations like `[1]`, `[2]` are required at the end of sentences derived from the text.

### **OUTPUT STRUCTURE**:
Please organize your response in the following Markdown format:

## **1. 요약 (Summary)**
- Provide a concise summary of the answer (2-3 sentences).

## **2. 상세 설명 (Detailed Explanation)**
- **정의 (Definition)**: What is the core concept?
- **주요 역할 및 원리 (Role & Mechanism)**: How does it work? Detailed explanation using the context.

## **3. 주요 구성 요소 및 특징 (Components & Features)**
- List key components, attributes, or constraints mentioned.
- Use bullet points.

## **4. 예시 또는 관련 흐름 (Examples or Related Flows)** (If applicable)
- Describe any specific examples, call flows, or use cases found in the text.
"""

    # 사용자 프롬프트 템플릿
    USER_PROMPT_TEMPLATE = """=== REFERENCE DOCUMENTS (Ranked by relevance) ===
{context}

=== QUESTION ===
{question}

=== GUIDELINES ===
- Answer in Korean.
- Follow the structure: Summary -> Detailed Explanation -> Components -> Examples.
- Keep technical terms in English but explain them.

=== ANSWER ===
"""

    # 쿼리 재작성 프롬프트 템플릿
    QUERY_REWRITE_TEMPLATE = """You are a search query optimizer. 
Original Question: "{question}"

Task: Rewrite the question into a better search query for a technical documentation search engine.
- Remove conversational filler (e.g., "Hi", "Can you tell me")
- Focus on key technical terms
- Expand abbreviations if ambiguous
- Include synonyms or related technical terms to broaden search coverage (e.g., "mobile" -> "mobile terminal", "UE", "device")
- Return ONLY the rewritten query text. Do not explain.

Rewritten Query:"""


# ============================================================================
# 통합 설정
# ============================================================================

class AppConfig:
    """애플리케이션 통합 설정."""
    
    # 하위 설정 인스턴스
    LOGGING = LoggingConfig()
    OLLAMA = OllamaConfig()
    LMSTUDIO = LMStudioConfig()
    LLM_BACKEND = LLMBackendConfig()
    VECTOR_STORE = VectorStoreConfig()
    CHUNKING = ChunkingConfig()
    RETRIEVER = RetrieverConfig()
    CACHE = CacheConfig()
    RAG = RAGConfig()
    CONVERSION = ConversionConfig()
    INDEXING = IndexingConfig()
    BATCH = BatchConfig()
    API = APIConfig()
    PROMPT = PromptConfig()
    
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
