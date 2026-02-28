# MarkItDown 사용 가이드 (Python 네이티브)

이 문서는 macOS/Linux에서 Python 네이티브 환경으로 MarkItDown RAG 서비스를 실행하는 방법을 설명합니다.

## 1. 사전 준비

- Python 3.10+
- 가상환경 도구(`venv`)
- LibreOffice (`.doc` 변환 필요 시)
- LM Studio (LLM 추론 서버, 항상 실행)
- OpenAI 호환 임베딩 서버 (Python 네이티브, 항상 실행)

## 2. 프로젝트 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 3. 환경 변수 설정

`.env.example`을 복사하여 `.env`를 생성하고 필요한 값으로 수정합니다.

```bash
cp .env.example .env
```

핵심 설정:

- `LLM_BACKEND_TYPE=lmstudio`
- `LMSTUDIO_BASE_URL=http://localhost:1234/api`
- `LMSTUDIO_LLM_MODEL=<LM Studio에 로드된 모델명>`
- `LMSTUDIO_EMBEDDING_SERVICE_URL=http://localhost:8001`
- `LMSTUDIO_EMBEDDING_MODEL=<임베딩 서버 모델명>`
- `MARKITDOWN_INPUT_DIR=./input`
- `MARKITDOWN_OUTPUT_DIR=./output`
- `VECTOR_STORE_DIR=./vector_store`
- `BATCH_STATE_DIR=./batch_state`

## 4. 서버 실행

```bash
python -m uvicorn app.converter:app --host 0.0.0.0 --port 8000 --loop asyncio
```

확인:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`

## 5. 기본 API 사용 예시

### 단일 파일 변환

```bash
curl -X POST "http://localhost:8000/convert" \
  -F "file=@./input/sample.docx" \
  -F "auto_index=true"
```

### 입력 폴더 일괄 변환

```bash
curl -X POST "http://localhost:8000/convert-folder" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "auto_index=true"
```

### 출력 폴더 인덱싱

```bash
curl -X POST "http://localhost:8000/index-folder" \
  -H "Content-Type: application/json" \
  -d '{"folder_path":"./output"}'
```

### 검색

```bash
curl "http://localhost:8000/search?query=5G%20Core&top_k=5"
```

### RAG 질의

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "5G Core의 주요 구성 요소는?",
    "top_k": 5,
    "temperature": 0.3,
    "max_tokens": 1024,
    "include_sources": true
  }'
```

## 6. 테스트 실행

```bash
python test_basic.py
python test_embedding.py
python test_vector_store.py
python test_retriever.py
python test_rag.py
python test_indexer.py
```

통합 시나리오는 아래 스크립트를 사용합니다.

```bash
bash comprehensive_test.sh
```

## 7. 디렉토리 규약

- `./input`: 원본 문서 입력
- `./output`: 변환된 Markdown
- `./vector_store`: 벡터 DB 저장소
- `./batch_state`: 배치 작업 상태
- `./debug`: 테스트/디버그 CSV 출력

## 8. 문제 해결

- `/health` 실패: LM Studio 또는 임베딩 서버 URL/포트 확인
- 임베딩 실패: `LMSTUDIO_EMBEDDING_SERVICE_URL`의 `/v1/models`, `/v1/embeddings` 확인
- `.doc` 변환 실패: LibreOffice 설치 및 PATH 확인