"""
RAG API 요청/응답을 위한 Pydantic 스키마.
"""

from typing import Optional, List
from pydantic import BaseModel, Field   # type: ignore


# ============================================================================
# 임베딩 스키마
# ============================================================================

class EmbeddingRequest(BaseModel):
    """임베딩 생성 요청."""
    text: str = Field(..., description="임베딩할 텍스트")
    model: Optional[str] = Field(None, description="모델명 (제공되지 않으면 기본값 사용)")


class EmbeddingResponse(BaseModel):
    """임베딩을 포함한 응답."""
    embedding: List[float] = Field(..., description="임베딩 벡터")
    dimension: int = Field(..., description="임베딩 차원")
    model: str = Field(..., description="임베딩에 사용된 모델")


# ============================================================================
# 문서 스키마
# ============================================================================

class DocumentMetadata(BaseModel):
    """인덱싱된 문서의 메타데이터."""
    source: str = Field(..., description="원본 파일 경로")
    chunk_id: int = Field(..., description="문서 내 청크 인덱스")
    total_chunks: int = Field(..., description="문서의 총 청크 수")
    created_at: str = Field(..., description="인덱싱 타임스탬프")


class DocumentChunk(BaseModel):
    """메타데이터를 포함한 문서 청크."""
    id: str = Field(..., description="고유 청크 ID")
    content: str = Field(..., description="청크 텍스트 내용")
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = Field(None, description="청크 임베딩 벡터")


class Document(BaseModel):
    """인덱싱된 문서 정보."""
    id: str = Field(..., description="문서 ID (파일명)")
    filename: str = Field(..., description="원본 파일명")
    total_chunks: int = Field(..., description="청크 개수")
    indexed_at: str = Field(..., description="인덱싱 타임스탬프")
    status: str = Field(..., description="인덱싱 상태 (indexed, failed)")


# ============================================================================
# RAG 스키마
# ============================================================================

class RetrievalResult(BaseModel):
    """단일 검색 결과."""
    content: str = Field(..., description="검색된 텍스트 내용")
    source: str = Field(..., description="원본 문서")
    chunk_id: int = Field(..., description="청크 ID")
    similarity_score: float = Field(..., description="유사도 점수 (0-1)")


class RAGRequest(BaseModel):
    """RAG 질의 요청."""
    query: str = Field(..., description="사용자 질의")
    top_k: Optional[int] = Field(None, description="검색 결과 개수")
    temperature: Optional[float] = Field(None, description="LLM 온도 파라미터")
    max_tokens: Optional[int] = Field(None, description="응답의 최대 토큰 수")
    include_sources: bool = Field(True, description="응답에 원본 문서 포함")


class RAGResponse(BaseModel):
    """RAG 시스템의 응답."""
    answer: str = Field(..., description="생성된 답변")
    sources: List[RetrievalResult] = Field(default_factory=list, description="검색된 원본")
    model: str = Field(..., description="사용된 LLM 모델")
    tokens_used: Optional[int] = Field(None, description="사용된 대략적 토큰 수")


# ============================================================================
# 인덱싱 스키마
# ============================================================================

class IndexRequest(BaseModel):
    """문서 인덱싱 요청."""
    filename: Optional[str] = Field(None, description="인덱싱할 특정 파일")
    force_reindex: bool = Field(False, description="이미 인덱싱됨에도 강제 재인덱싱")


class IndexResponse(BaseModel):
    """인덱싱 작업의 응답."""
    total_files: int = Field(..., description="처리된 총 파일 수")
    indexed_files: int = Field(..., description="성공적으로 인덱싱된 파일 수")
    failed_files: int = Field(..., description="인덱싱 실패 파일 수")
    total_chunks: int = Field(..., description="생성된 총 청크 수")
    files: List[dict] = Field(..., description="파일별 상세 정보")


class IndexStatus(BaseModel):
    """인덱싱된 문서의 상태."""
    total_documents: int = Field(..., description="인덱싱된 총 문서 수")
    total_chunks: int = Field(..., description="총 텍스트 청크 수")
    vector_store_size: str = Field(..., description="벡터 저장소 크기")
    last_indexed: Optional[str] = Field(None, description="마지막 인덱싱 타임스탬프")


# ============================================================================
# 검색 스키마
# ============================================================================

class SearchRequest(BaseModel):
    """의미론적 검색 요청."""
    query: str = Field(..., description="검색 질의")
    top_k: Optional[int] = Field(None, description="결과 개수")
    similarity_threshold: Optional[float] = Field(None, description="최소 유사도 점수")


class SearchResponse(BaseModel):
    """의미론적 검색의 응답."""
    results: List[RetrievalResult] = Field(..., description="검색 결과")
    query_embedding_dim: int = Field(..., description="질의 임베딩 차원")
    total_results: int = Field(..., description="발견된 총 결과 수")


# ============================================================================
# 헬스 체크 스키마
# ============================================================================

class HealthResponse(BaseModel):
    """헬스 체크 응답."""
    status: str = Field(..., description="헬스 상태")
    ollama_available: bool = Field(..., description="Ollama 서버 가용성")
    vector_store_ready: bool = Field(..., description="벡터 저장소 가용성")
    models: dict = Field(..., description="사용 가능한 모델 정보")



# ============================================================================
# 에러 스키마
# ============================================================================

class ErrorResponse(BaseModel):
    """에러 응답."""
    detail: str = Field(..., description="에러 메시지")
    error_code: Optional[str] = Field(None, description="에러 코드")


# ============================================================================
# 파일 변환 스키마
# ============================================================================

class ConversionFileResult(BaseModel):
    """개별 파일 변환 결과."""
    input: str = Field(..., description="입력 파일명")
    status: str = Field(..., description="변환 상태 (success, failed)")
    duration: float = Field(..., description="변환 소요 시간 (초)")
    output: Optional[str] = Field(None, description="출력 파일명")
    reason: Optional[str] = Field(None, description="실패 사유")


class BatchConversionResult(BaseModel):
    """배치 파일 변환 결과."""
    total_files: int = Field(..., description="처리한 총 파일 수")
    converted_files: int = Field(..., description="성공적으로 변환된 파일 수")
    failed_files: int = Field(..., description="변환 실패한 파일 수")
    total_duration: float = Field(..., description="총 변환 시간 (초)")
    files: List[ConversionFileResult] = Field(..., description="파일별 변환 결과")
    message: str = Field(..., description="결과 메시지")


class SupportedFormatsResponse(BaseModel):
    """지원 파일 형식 응답."""
    formats: List[str] = Field(..., description="지원 파일 형식 목록")
    count: int = Field(..., description="지원 형식 개수")
