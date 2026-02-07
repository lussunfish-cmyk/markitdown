# schemas.py 구현

## 개요

RAG API의 요청/응답 데이터 구조를 정의하는 Pydantic 스키마 모듈입니다. FastAPI와 통합되어 자동 검증, 문서화, 직렬화/역직렬화를 제공합니다.

## 파일 경로

```
markitdown/app/schemas.py
```

## 주요 구성 요소

### 1. 임베딩 스키마

#### EmbeddingRequest

임베딩 생성 요청 스키마입니다.

```python
class EmbeddingRequest(BaseModel):
    """임베딩 생성 요청."""
    text: str = Field(..., description="임베딩할 텍스트")
    model: Optional[str] = Field(None, description="모델명 (제공되지 않으면 기본값 사용)")
```

**필드:**
- `text` (필수): 임베딩을 생성할 텍스트
- `model` (선택): 사용할 모델명 (미지정 시 config의 기본 모델 사용)

#### EmbeddingResponse

임베딩 생성 응답 스키마입니다.

```python
class EmbeddingResponse(BaseModel):
    """임베딩을 포함한 응답."""
    embedding: List[float] = Field(..., description="임베딩 벡터")
    dimension: int = Field(..., description="임베딩 차원")
    model: str = Field(..., description="임베딩에 사용된 모델")
```

**필드:**
- `embedding`: 생성된 임베딩 벡터 (float 리스트)
- `dimension`: 벡터의 차원 수
- `model`: 사용된 모델명

### 2. 문서 스키마

#### DocumentMetadata

인덱싱된 문서의 메타데이터를 나타냅니다.

```python
class DocumentMetadata(BaseModel):
    """인덱싱된 문서의 메타데이터."""
    source: str = Field(..., description="원본 파일 경로")
    chunk_id: int = Field(..., description="문서 내 청크 인덱스")
    total_chunks: int = Field(..., description="문서의 총 청크 수")
    created_at: str = Field(..., description="인덱싱 타임스탬프")
```

#### DocumentChunk

메타데이터를 포함한 문서 청크입니다.

```python
class DocumentChunk(BaseModel):
    """메타데이터를 포함한 문서 청크."""
    id: str = Field(..., description="고유 청크 ID")
    content: str = Field(..., description="청크 텍스트 내용")
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = Field(None, description="청크 임베딩 벡터")
```

**필드:**
- `id`: 청크의 고유 식별자 (예: `doc1_chunk_0`)
- `content`: 청크의 실제 텍스트 내용
- `metadata`: 청크의 메타데이터
- `embedding`: 청크의 임베딩 벡터 (선택)

#### Document

인덱싱된 문서 정보입니다.

```python
class Document(BaseModel):
    """인덱싱된 문서 정보."""
    id: str = Field(..., description="문서 ID (파일명)")
    filename: str = Field(..., description="원본 파일명")
    total_chunks: int = Field(..., description="청크 개수")
    indexed_at: str = Field(..., description="인덱싱 타임스탬프")
    status: str = Field(..., description="인덱싱 상태 (indexed, failed)")
```

### 3. RAG 스키마

#### RetrievalResult

단일 검색 결과를 나타냅니다.

```python
class RetrievalResult(BaseModel):
    """단일 검색 결과."""
    content: str = Field(..., description="검색된 텍스트 내용")
    source: str = Field(..., description="원본 문서")
    chunk_id: int = Field(..., description="청크 ID")
    similarity_score: float = Field(..., description="유사도 점수 (0-1)")
```

**필드:**
- `content`: 검색된 텍스트 내용
- `source`: 원본 문서 파일명
- `chunk_id`: 청크 인덱스
- `similarity_score`: 쿼리와의 유사도 점수 (0~1)

#### RAGRequest

RAG 질의 요청 스키마입니다.

```python
class RAGRequest(BaseModel):
    """RAG 질의 요청."""
    query: str = Field(..., description="사용자 질의")
    top_k: Optional[int] = Field(None, description="검색 결과 개수")
    temperature: Optional[float] = Field(None, description="LLM 온도 파라미터")
    max_tokens: Optional[int] = Field(None, description="응답의 최대 토큰 수")
    include_sources: bool = Field(True, description="응답에 원본 문서 포함")
```

**필드:**
- `query` (필수): 사용자의 질의 텍스트
- `top_k` (선택): 검색할 문서 개수 (미지정 시 config 기본값)
- `temperature` (선택): LLM 생성 온도
- `max_tokens` (선택): 최대 생성 토큰 수
- `include_sources`: 응답에 검색된 원본 문서 포함 여부

**사용 예시:**

```python
request = RAGRequest(
    query="5G 기술의 특징은?",
    top_k=5,
    temperature=0.7,
    include_sources=True
)
```

#### RAGResponse

RAG 시스템의 응답 스키마입니다.

```python
class RAGResponse(BaseModel):
    """RAG 시스템의 응답."""
    answer: str = Field(..., description="생성된 답변")
    sources: List[RetrievalResult] = Field(default_factory=list, description="검색된 원본")
    model: str = Field(..., description="사용된 LLM 모델")
    tokens_used: Optional[int] = Field(None, description="사용된 대략적 토큰 수")
```

**필드:**
- `answer`: LLM이 생성한 답변
- `sources`: 참조한 문서 청크들
- `model`: 사용된 LLM 모델명
- `tokens_used`: 사용된 토큰 수 (추정치)

### 4. 인덱싱 스키마

#### IndexRequest

문서 인덱싱 요청 스키마입니다.

```python
class IndexRequest(BaseModel):
    """문서 인덱싱 요청."""
    filename: Optional[str] = Field(None, description="인덱싱할 특정 파일")
    force_reindex: bool = Field(False, description="이미 인덱싱됨에도 강제 재인덱싱")
```

**필드:**
- `filename` (선택): 특정 파일만 인덱싱 (미지정 시 전체)
- `force_reindex`: 기존 인덱싱 무시하고 재인덱싱 여부

#### IndexResponse

인덱싱 작업의 응답 스키마입니다.

```python
class IndexResponse(BaseModel):
    """인덱싱 작업의 응답."""
    total_files: int = Field(..., description="처리한 총 파일 수")
    indexed_files: int = Field(..., description="성공적으로 인덱싱된 파일 수")
    failed_files: int = Field(..., description="인덱싱 실패한 파일 수")
    total_chunks: int = Field(..., description="생성된 총 청크 수")
    files: List[dict] = Field(..., description="파일별 상세 정보")
```

#### IndexStatus

인덱싱된 문서의 상태 정보입니다.

```python
class IndexStatus(BaseModel):
    """인덱싱된 문서의 상태."""
    total_documents: int = Field(..., description="인덱싱된 총 문서 수")
    total_chunks: int = Field(..., description="총 텍스트 청크 수")
    vector_store_size: str = Field(..., description="벡터 저장소 크기")
    last_indexed: Optional[str] = Field(None, description="마지막 인덱싱 타임스탬프")
```

### 5. 검색 스키마

#### SearchRequest

의미론적 검색 요청 스키마입니다.

```python
class SearchRequest(BaseModel):
    """의미론적 검색 요청."""
    query: str = Field(..., description="검색 질의")
    top_k: Optional[int] = Field(None, description="결과 개수")
    similarity_threshold: Optional[float] = Field(None, description="최소 유사도 점수")
```

#### SearchResponse

의미론적 검색의 응답 스키마입니다.

```python
class SearchResponse(BaseModel):
    """의미론적 검색의 응답."""
    results: List[RetrievalResult] = Field(..., description="검색 결과")
    query_embedding_dim: int = Field(..., description="질의 임베딩 차원")
    total_results: int = Field(..., description="발견된 총 결과 수")
```

### 6. 헬스 체크 스키마

#### HealthResponse

시스템 헬스 체크 응답입니다.

```python
class HealthResponse(BaseModel):
    """헬스 체크 응답."""
    status: str = Field(..., description="헬스 상태")
    ollama_available: bool = Field(..., description="Ollama 서버 가용성")
    vector_store_ready: bool = Field(..., description="벡터 저장소 가용성")
    models: dict = Field(..., description="사용 가능한 모델 정보")
```

### 7. 에러 스키마

#### ErrorResponse

API 에러 응답 스키마입니다.

```python
class ErrorResponse(BaseModel):
    """에러 응답."""
    detail: str = Field(..., description="에러 메시지")
    error_code: Optional[str] = Field(None, description="에러 코드")
```

### 8. 파일 변환 스키마

#### ConversionFileResult

개별 파일 변환 결과입니다.

```python
class ConversionFileResult(BaseModel):
    """개별 파일 변환 결과."""
    input: str = Field(..., description="입력 파일명")
    status: str = Field(..., description="변환 상태 (success, failed)")
    duration: float = Field(..., description="변환 소요 시간 (초)")
    output: Optional[str] = Field(None, description="출력 파일명")
    reason: Optional[str] = Field(None, description="실패 사유")
```

#### BatchConversionResult

배치 파일 변환 결과입니다.

```python
class BatchConversionResult(BaseModel):
    """배치 파일 변환 결과."""
    total_files: int = Field(..., description="처리한 총 파일 수")
    converted_files: int = Field(..., description="성공적으로 변환된 파일 수")
    failed_files: int = Field(..., description="변환 실패한 파일 수")
    total_duration: float = Field(..., description="총 변환 시간 (초)")
    files: List[ConversionFileResult] = Field(..., description="파일별 변환 결과")
    message: str = Field(..., description="결과 메시지")
```

#### SupportedFormatsResponse

지원 파일 형식 응답입니다.

```python
class SupportedFormatsResponse(BaseModel):
    """지원 파일 형식 응답."""
    formats: List[str] = Field(..., description="지원 파일 형식 목록")
    count: int = Field(..., description="지원 형식 개수")
```

## FastAPI 통합

### 자동 검증

FastAPI는 요청 데이터를 자동으로 검증합니다:

```python
@app.post("/query", response_model=RAGResponse)
async def query_rag(request: RAGRequest):
    # request는 이미 검증되고 파싱됨
    return await process_rag_query(request)
```

### 자동 문서화

스키마는 OpenAPI 문서에 자동으로 포함됩니다:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 타입 안전성

Pydantic은 런타임 타입 검증을 제공합니다:

```python
# 유효한 요청
request = RAGRequest(query="질문", top_k=5)  # ✅

# 유효하지 않은 요청 (ValidationError 발생)
request = RAGRequest(query=123, top_k="five")  # ❌
```

## 사용 예시

### 요청 생성

```python
from schemas import RAGRequest, SearchRequest

# RAG 요청
rag_req = RAGRequest(
    query="5G의 장점은?",
    top_k=3,
    temperature=0.7,
    include_sources=True
)

# 검색 요청
search_req = SearchRequest(
    query="VoLTE 설정",
    top_k=5,
    similarity_threshold=0.7
)
```

### 응답 처리

```python
from schemas import RAGResponse

# FastAPI에서 자동으로 직렬화됨
response = RAGResponse(
    answer="5G의 주요 장점은...",
    sources=[
        RetrievalResult(
            content="5G 기술 설명...",
            source="5G.md",
            chunk_id=0,
            similarity_score=0.92
        )
    ],
    model="gemma2",
    tokens_used=350
)
```

### JSON 변환

```python
# Python 객체 → JSON
json_data = response.model_dump_json()

# JSON → Python 객체
response = RAGResponse.model_validate_json(json_data)
```

## 설계 원칙

1. **명확한 네이밍**: 각 스키마의 목적이 이름에서 명확히 드러남
2. **타입 안전성**: Pydantic을 통한 강력한 타입 검증
3. **문서화**: Field의 description으로 자동 문서 생성
4. **Optional 필드**: 선택적 필드는 Optional로 명시
5. **기본값 제공**: default_factory 등을 활용한 합리적 기본값

## 확장 가이드

새로운 API를 추가할 때:

1. 요청 스키마 정의
2. 응답 스키마 정의
3. Field description 작성
4. 적절한 타입과 검증 규칙 설정

```python
class NewFeatureRequest(BaseModel):
    """새 기능 요청."""
    param1: str = Field(..., description="필수 파라미터")
    param2: Optional[int] = Field(None, description="선택 파라미터")

class NewFeatureResponse(BaseModel):
    """새 기능 응답."""
    result: str = Field(..., description="처리 결과")
    status: str = Field(..., description="처리 상태")
```
