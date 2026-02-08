# Schemas 구현

## 개요

FastAPI 요청 및 응답, 그리고 내부 데이터 교환에 사용되는 Pydantic 모델(데이터 스키마)을 정의한 모듈입니다. 데이터 유효성 검사와 API 문서화(Swagger/OpenAPI)를 지원합니다.

## 파일 경로

```
markitdown/app/schemas.py
markitdown/app/schemas_eval.py
```

## 주요 스키마

- **문서 및 임베딩**: `Document`, `DocumentChunk`, `DocumentMetadata`, `EmbeddingRequest/Response`.
- **RAG**: `RAGRequest` (질문, 옵션), `RAGResponse` (답변, 출처), `RetrievalResult` (검색된 문서 정보).
- **인덱싱**: `IndexRequest`, `IndexResponse`, `IndexStatus`.
- **변환**: `ConversionFileResult`, `BatchConversionResult`.
- **배치 처리**: `BatchJobResponse`, `BatchInfo`, `BatchFileStatus`.
- **평가 (schemas_eval)**: `TestsetGenerateRequest`, `EvaluationRequest`, `EvaluationResponse`.

## 특징
- 각 필드에 `Field(..., description="...")`를 사용하여 API 문서에 상세한 설명을 제공합니다.