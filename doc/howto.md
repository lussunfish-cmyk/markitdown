# MarkItDown 사용 가이드 (Python 네이티브)

이 문서는 macOS/Linux에서 Python 네이티브 환경으로 MarkItDown RAG 서비스를 실행하는 방법을 설명합니다.

## 1. 사전 준비

- Python 3.10+
- 가상환경 도구(`venv`)
- LibreOffice (`.doc` 변환 필요 시)
  ```bash
  brew install --cask libreoffice
  ```
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

## 4. 임베딩 서비스 설정 (선택사항)

**벡터 검색/인덱싱이 필요한 경우**

임베딩 서비스가 없으면 파일 변환은 정상이지만, `auto_index=true`로 요청할 때 인덱싱이 실패합니다.

#### 옵션 1: 임베딩 서비스 없이 사용
- 파일 변환만 필요한 경우: `auto_index=false` 또는 생략
- 나중에 `/index-folder` API로 인덱싱

#### 옵션 2: 임베딩 서비스 실행
별도의 Python 임베딩 서비스(예: Ollama, FastEmbed, vLLM 등)를 실행하고 URL을 설정:

```bash
# 예: Ollama (또는 다른 embedding service)
LMSTUDIO_EMBEDDING_SERVICE_URL=http://localhost:8001
LMSTUDIO_EMBEDDING_MODEL=mxbai-embed-large-v1
```

## 5. 서버 실행

```bash
python -m uvicorn app.converter:app --host 0.0.0.0 --port 8000 --loop asyncio
```

확인:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`

## 6. 기본 API 사용 예시

### 단일 파일 변환 (임베딩 서비스 필요 없음)

```bash
curl -X POST "http://localhost:8000/convert" \
  -F "file=@./input/sample.docx"
```

### 단일 파일 변환 + 자동 인덱싱 (임베딩 서비스 필요)

```bash
curl -X POST "http://localhost:8000/convert" \
  -F "file=@./input/sample.docx" \
  -F "auto_index=true"
```

### 입력 폴더 일괄 변환 (인덱싱 없음)

```bash
curl -X POST "http://localhost:8000/convert-folder" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "auto_index=false"
```

### 입력 폴더 일괄 변환 + 자동 인덱싱 (임베딩 서비스 필요)

```bash
curl -X POST "http://localhost:8000/convert-folder" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "auto_index=true"
```

### 출력 폴더 인덱싱 (임베딩 서비스 필요)

```bash
curl -X POST "http://localhost:8000/index-folder" \
  -H "Content-Type: application/json" \
  -d '{"folder_path":"./output"}'
```

### 검색 (임베딩 서비스 필요)

```bash
curl "http://localhost:8000/search?query=5G%20Core&top_k=5"
```

### RAG 질의 (임베딩 서비스 필요)

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

## 7. 테스트 실행

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

## 8. 디렉토리 규약

- `./input`: 원본 문서 입력
- `./output`: 변환된 Markdown
- `./vector_store`: 벡터 DB 저장소
- `./batch_state`: 배치 작업 상태
- `./debug`: 테스트/디버그 CSV 출력

## 9. 문제 해결

### LM Studio 연결 실패
- `/health` 실패: LM Studio가 `http://localhost:1234/api`에서 실행 중인지 확인
- 명령: `curl http://localhost:1234/api/v1/models`

### 임베딩 서비스 오류
**증상**: "인덱싱 실패: 임베딩 서비스를 사용할 수 없습니다"

**해결**:
1. **임베딩 서비스 없이 사용** (변환만 필요): `auto_index=false`로 요청
2. **임베딩 서비스 구성**: 별도 서비스 실행 후 `.env` 수정:
   ```bash
   LMSTUDIO_EMBEDDING_SERVICE_URL=http://localhost:8001
   LMSTUDIO_EMBEDDING_MODEL=mxbai-embed-large-v1
   ```
3. **서비스 확인**: `curl http://localhost:8001/v1/models`

### .doc 변환 실패
- LibreOffice 설치 확인:
  ```bash
  soffice --version
  # 또는 (macOS)
  /Applications/LibreOffice.app/Contents/MacOS/soffice --version
  ```