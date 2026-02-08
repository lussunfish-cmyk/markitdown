# schemas.py 구현

## 개요

RAG API의 모든 요청 및 응답 데이터 구조를 정의하는 Pydantic 스키마 모듈입니다. 데이터 검증(Validation)과 API 문서화(OpenAPI/Swagger)를 위해 사용됩니다.

## 파일 경로

```
markitdown/app/schemas.py
```

## 주요 스키마

### 1. 임베딩 (Embedding)
- **EmbeddingRequest**: 텍스트와 모델 옵션.
- **EmbeddingResponse**: 생성된 벡터(float 리스트), 차원, 사용 모델.

### 2. 문서 및 청크 (Document & Chunk)
- **DocumentMetadata**: 청크 인덱스, 총 청크 수, 생성일, 섹션 제목 등을 포함.
- **DocumentChunk**: 청크 ID, 내용, 메타데이터, 임베딩(Optional).
- **Document**: 파일 단위의 인덱싱 정보 (filename, status, total_chunks 등).

### 3. 검색 및 RAG (Retrieval & RAG)
- **RetrievalResult**: 검색된 텍스트, 소스 파일, 유사도 점수.
- **RAGRequest**: 질의(query), Top-K, Temperature, Max Tokens 등 생성 옵션.
- **RAGResponse**: 생성된 답변(answer), 참조 소스(sources), 사용 모델, 토큰 사용량.
- **SearchRequest**: 의미론적 검색 요청 (Query, threshold).
- **SearchResponse**: 검색 결과 리스트 및 메타 정보.

### 4. 인덱싱 (Indexing)
- **IndexRequest**: 특정 파일 또는 강제 재인덱싱 요청.
- **IndexResponse**: 인덱싱 작업 결과 통계 (성공/실패 파일 수, 청크 수).
- **IndexStatus**: 현재 시스템의 인덱싱 상태 (총 문서 수, 벡터 스토어 정보).

### 5. 파일 변환 (Conversion)
- **ConversionFileResult**: 개별 파일 변환 상태 및 결과 경로.
- **BatchConversionResult**: 배치 변환 작업의 요약 (총 파일, 성공/실패 등).
- **SupportedFormatsResponse**: 지원하는 파일 확장자 목록.

### 6. 배치 처리 (Batch Processing)
- **BatchFileStatus**: 배치 내 개별 파일의 상세 상태 (변환 경로, 인덱싱 여부, 처리 시간 등).
- **BatchInfo**: 배치 그룹 단위의 상태 정보.
- **BatchJobResponse**: 전체 배치 작업의 진행 상황 및 결과.

### 7. 기타
- **HealthResponse**: 시스템 상태, Ollama 및 벡터 스토어 연결 상태.
- **ErrorResponse**: 표준 에러 응답 포맷.
- **QueryStreamChunk**: 스트리밍 응답을 위한 청크 구조 (타입, 내용, 소스 등).
- **SearchResult**: 단순화된 검색 결과 구조.
