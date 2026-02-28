# MarkItDown 아키텍처 (Python 네이티브)

## 1. 개요

MarkItDown은 문서 변환 → 청킹/임베딩 → 벡터 인덱싱 → 검색/RAG 응답까지를 제공하는 FastAPI 기반 서비스입니다.

운영 전제:

- LLM 추론: LM Studio 서버
- 임베딩: 별도 Python 네이티브 임베딩 서버(OpenAI 호환 API)
- 애플리케이션: Python 네이티브 프로세스로 실행

## 2. 주요 컴포넌트

- `app/converter.py`: FastAPI 엔트리포인트 및 API 라우트
- `app/indexer.py`: 문서 인덱싱 파이프라인
- `app/embedding.py`: 텍스트 청킹 및 임베딩 호출
- `app/retriever.py`: 하이브리드 검색/리랭킹
- `app/rag.py`: 컨텍스트 구성 및 답변 생성
- `app/lm_studio_client.py`: LM Studio + 임베딩 서비스 통신 클라이언트
- `app/vector_store.py`: Chroma 기반 벡터 저장소
- `app/config.py`: 환경 변수 중심 설정

## 3. 데이터 흐름

1. 입력 문서를 `./input`에 배치
2. 변환 API가 문서를 Markdown으로 변환 후 `./output` 저장
3. 인덱서가 문서를 청킹하고 임베딩 서버로 벡터 생성 요청
4. 벡터를 `./vector_store`에 저장
5. 질의 시 검색기로 관련 청크를 찾고 LM Studio로 답변 생성

## 4. 디렉토리/상태 저장

- `./input`: 원본 입력
- `./output`: 변환 결과
- `./vector_store`: 벡터 DB 및 인덱스 상태
- `./batch_state`: 배치 작업 상태
- `./tiktoken_cache`: 토크나이저 캐시

## 5. 외부 의존 서비스

### LLM 서버 (LM Studio)

- URL: `LMSTUDIO_BASE_URL` (예: `http://localhost:1234/api`)
- API: OpenAI 호환 Chat Completions

### 임베딩 서버 (Python 네이티브)

- URL: `LMSTUDIO_EMBEDDING_SERVICE_URL` (예: `http://localhost:8001`)
- 필수 엔드포인트:
  - `GET /v1/models`
  - `POST /v1/embeddings`

## 6. 설정 전략

모든 런타임 설정은 환경 변수로 주입합니다.

대표 기본값:

- `LLM_BACKEND_TYPE=lmstudio`
- `MARKITDOWN_INPUT_DIR=./input`
- `MARKITDOWN_OUTPUT_DIR=./output`
- `VECTOR_STORE_DIR=./vector_store`
- `BATCH_STATE_DIR=./batch_state`

## 7. 실행

```bash
python -m uvicorn app.converter:app --host 0.0.0.0 --port 8000 --loop asyncio
```

## 8. 확장 포인트

- `LLMClient` 추상 인터페이스 기반 백엔드 확장
- 벡터 저장소 타입 확장(`chroma`, `faiss`)
- 리랭커 모델 교체/비활성화
- 배치 처리 정책(크기/타임아웃) 조정