# MarkItDown 통합 API 가이드

FastAPI 기반 마크다운 변환 + RAG 통합 서비스입니다.

## 🎯 핵심 기능

### 1. 변환 (Conversion)
- **단일 파일 변환**: PDF, DOCX, PPTX 등 → Markdown
- **폴더 배치 변환**: 여러 파일 일괄 처리
- **대용량 배치 처리**: 수백~수천 개 파일 체크포인트 기반 처리

### 2. 인덱싱 (Indexing)
- **자동 인덱싱**: 변환 시 자동으로 벡터 DB 저장 (auto_index)
- **수동 인덱싱**: 기존 마크다운 파일 인덱싱
- **ChromaDB**: 벡터 저장소로 관리

### 3. RAG (검색 및 질의응답)
- **문서 검색**: 유사 문서 자동 검색
- **AI 답변 생성**: Ollama gemma2 모델로 답변
- **출처 추적**: 답변 근거 문서 제공

## 📦 지원 파일 형식

- **문서**: PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS
- **텍스트**: TXT, CSV, JSON, XML, HTML, HTM, MD
- **이미지**: PNG, JPG, JPEG, GIF, BMP, TIFF
- **미디어**: WAV, MP3, M4A, FLAC
- **압축**: ZIP, EPUB


## 🚀 빠른 시작

### Docker Compose로 실행

```bash
# 컨테이너 빌드 및 실행
docker compose up --build

# 백그라운드 실행
docker compose up -d --build
```

서비스 확인:
```bash
# 헬스 체크
curl http://localhost:8000/health

# API 문서 (브라우저)
http://localhost:8000/docs
```

### 데이터 영속성

Docker Compose는 다음 볼륨을 자동 마운트합니다:

```yaml
volumes:
  - ./input:/app/input           # 입력 파일
  - ./output:/app/output         # 변환된 Markdown
  - ./vector_store:/app/vector_store    # ChromaDB 벡터 저장소
  - ./batch_state:/app/batch_state      # 배치 작업 상태
```

---

## 📡 API 엔드포인트

### 전체 목록

| 카테고리 | 메서드 | 엔드포인트 | 설명 |
|---------|--------|-----------|------|
| **변환** | POST | `/convert` | 단일 파일 변환 |
| **변환** | POST | `/convert-folder` | 폴더 전체 변환 |
| **변환** | POST | `/convert-batch` | 파일 업로드 배치 변환 |
| **배치** | POST | `/batch/folder` | 서버 폴더 배치 처리 |
| **배치** | GET | `/batch/{batch_id}` | 배치 상태 조회 |
| **배치** | GET | `/batch` | 전체 배치 목록 |
| **배치** | DELETE | `/batch/{batch_id}` | 배치 삭제 |
| **인덱싱** | POST | `/index` | 단일 파일 인덱싱 |
| **인덱싱** | POST | `/index-folder` | 폴더 전체 인덱싱 |
| **인덱싱** | GET | `/documents` | 인덱싱된 문서 목록 |
| **RAG** | POST | `/query` | RAG 질의응답 |
| **RAG** | GET | `/search` | 유사 문서 검색 |
| **유틸** | GET | `/health` | 헬스 체크 |
| **유틸** | GET | `/supported-formats` | 지원 파일 형식 |

---

## 📄 변환 API (Conversion)

### 1. POST /convert - 단일 파일 변환

**목적**: 하나의 파일을 업로드하여 Markdown으로 변환합니다.

**주요 기능**:
- 파일 업로드 후 즉시 변환
- 자동 인덱싱 옵션 (auto_index)
- 변환된 파일 자동 저장

**요청**:
```bash
curl -X POST "http://localhost:8000/convert" \
  -F "file=@document.pdf" \
  -F "auto_index=true"
```

**파라미터**:
- `file` (required): 변환할 파일
- `auto_index` (optional): 자동 인덱싱 여부 (기본값: false)

**응답 예시**:
```json
{
  "filename": "document.md",
  "message": "Converted successfully",
  "indexed": true
}
```

**사용 시나리오**:
- PDF 보고서를 Markdown으로 변환하여 편집
- 변환 + 인덱싱을 한 번에 수행 (auto_index=true)

---

### 2. POST /convert-folder - 폴더 전체 변환

**목적**: 서버의 input 폴더에 있는 모든 파일을 일괄 변환합니다.

**주요 기능**:
- 폴더 내 모든 지원 파일 자동 감지
- 순차 처리
- 자동 인덱싱 옵션

**요청**:
```bash
curl -X POST "http://localhost:8000/convert-folder?auto_index=true"
```

**파라미터**:
- `auto_index` (optional): 자동 인덱싱 여부 (기본값: false)

**응답 예시**:
```json
{
  "total_files": 5,
  "converted_files": 4,
  "failed_files": 1,
  "files": [
    {
      "input": "report.pdf",
      "output": "report.md",
      "status": "success",
      "indexed": true
    },
    {
      "input": "presentation.pptx",
      "output": "presentation.md",
      "status": "success",
      "indexed": true
    },
    {
      "input": "corrupted.pdf",
      "status": "failed",
      "reason": "Failed to read PDF"
    }
  ],
  "message": "Batch conversion complete: 4 succeeded, 1 failed"
}
```

**사용 시나리오**:
```bash
# 1. 파일을 input 폴더에 복사
cp /path/to/files/*.pdf ./input/

# 2. 폴더 변환 실행
curl -X POST "http://localhost:8000/convert-folder?auto_index=true"

# 3. 결과 확인
ls ./output/
```

---

### 3. POST /convert-batch - 파일 업로드 배치 변환

**목적**: 여러 파일을 한 번에 업로드하여 배치 변환합니다.

**주요 기능**:
- 다중 파일 업로드
- 비동기 배치 처리
- 진행률 추적

**요청**:
```bash
curl -X POST "http://localhost:8000/convert-batch" \
  -F "files=@file1.pdf" \
  -F "files=@file2.docx" \
  -F "files=@file3.pptx" \
  -F "batch_size=100" \
  -F "auto_index=true"
```

**파라미터**:
- `files` (required): 업로드할 파일들
- `batch_size` (optional): 배치 크기 (기본값: 100)
- `auto_index` (optional): 자동 인덱싱 여부

**응답 예시**:
```json
{
  "batch_id": "batch-20260207-143052-abc123",
  "total_files": 3,
  "total_batches": 1,
  "status": "completed",
  "progress_percentage": 100.0,
  "batches": [
    {
      "batch_num": 1,
      "total_files": 3,
      "completed": 3,
      "failed": 0,
      "status": "completed"
    }
  ]
}
```

---

## 🔄 배치 처리 API (Batch Processing)

### 1. POST /batch/folder - 서버 폴더 배치 처리

**목적**: 대용량 파일(수백~수천 개)을 체크포인트 기반으로 처리합니다.

**주요 기능**:
- JSON 기반 상태 저장 (재시작 가능)
- 100개 단위 배치 분할
- 파일별 상태 추적
- 중단/재시작 지원

**요청**:
```bash
curl -X POST "http://localhost:8000/batch/folder" \
  -H "Content-Type: application/json" \
  -d '{
    "folder_path": "/app/input",
    "batch_size": 100,
    "auto_index": true
  }'
```

**요청 Body**:
```json
{
  "folder_path": "/app/input",
  "batch_size": 100,
  "auto_index": true
}
```

**응답 예시**:
```json
{
  "batch_id": "batch-20260207-150030-def456",
  "total_files": 300,
  "total_batches": 3,
  "status": "processing",
  "progress_percentage": 65.3,
  "batches": [
    {
      "batch_num": 1,
      "total_files": 100,
      "completed": 100,
      "failed": 0,
      "status": "completed"
    },
    {
      "batch_num": 2,
      "total_files": 100,
      "completed": 96,
      "failed": 0,
      "status": "processing"
    },
    {
      "batch_num": 3,
      "total_files": 100,
      "completed": 0,
      "failed": 0,
      "status": "pending"
    }
  ]
}
```

**사용 시나리오**:
```bash
# 1. 대량 파일을 input 폴더에 복사 (예: 1000개)
cp /archive/*.pdf ./input/

# 2. 배치 처리 시작
curl -X POST "http://localhost:8000/batch/folder" \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "/app/input", "batch_size": 100, "auto_index": true}'

# 응답: batch_id 저장
# {"batch_id": "batch-20260207-150030-def456", ...}

# 3. 진행률 모니터링 (별도 터미널)
watch -n 5 'curl http://localhost:8000/batch/batch-20260207-150030-def456'
```

---

### 2. GET /batch/{batch_id} - 배치 상태 조회

**목적**: 진행 중이거나 완료된 배치 작업의 상태를 조회합니다.

**요청**:
```bash
curl "http://localhost:8000/batch/batch-20260207-150030-def456"
```

**응답 예시**:
```json
{
  "batch_id": "batch-20260207-150030-def456",
  "total_files": 300,
  "total_batches": 3,
  "status": "completed",
  "progress_percentage": 100.0,
  "started_at": "2026-02-07T15:00:30",
  "completed_at": "2026-02-07T15:25:18",
  "batches": [
    {
      "batch_num": 1,
      "total_files": 100,
      "completed": 98,
      "failed": 2,
      "status": "completed",
      "files": [
        {
          "filename": "file1.pdf",
          "status": "completed",
          "converted_path": "output/file1.md",
          "indexed": true,
          "duration": 5.2
        },
        {
          "filename": "file2.pdf",
          "status": "failed",
          "error": "Invalid PDF format"
        }
      ]
    }
  ]
}
```

---

### 3. GET /batch - 전체 배치 목록

**목적**: 저장된 모든 배치 작업 목록을 조회합니다.

**요청**:
```bash
curl "http://localhost:8000/batch"
```

**응답 예시**:
```json
{
  "total": 3,
  "batches": [
    {
      "batch_id": "batch-20260207-143052-abc123",
      "status": "completed",
      "total_files": 50,
      "progress_percentage": 100.0,
      "started_at": "2026-02-07T14:30:52"
    },
    {
      "batch_id": "batch-20260207-150030-def456",
      "status": "processing",
      "total_files": 300,
      "progress_percentage": 65.3,
      "started_at": "2026-02-07T15:00:30"
    },
    {
      "batch_id": "batch-20260207-153015-ghi789",
      "status": "pending",
      "total_files": 150,
      "progress_percentage": 0.0,
      "started_at": "2026-02-07T15:30:15"
    }
  ]
}
```

---

### 4. DELETE /batch/{batch_id} - 배치 삭제

**목적**: 완료되거나 실패한 배치 작업을 삭제합니다.

**요청**:
```bash
curl -X DELETE "http://localhost:8000/batch/batch-20260207-143052-abc123"
```

**응답 예시**:
```json
{
  "message": "Batch batch-20260207-143052-abc123 deleted successfully"
}
```

---

## 📚 인덱싱 API (Indexing)

### 1. POST /index - 단일 파일 인덱싱

**목적**: 이미 변환된 Markdown 파일을 벡터 DB에 저장합니다.

**주요 기능**:
- 청킹 (512자 단위, 128 오버랩)
- Ollama 임베딩 (mxbai-embed-large)
- ChromaDB 저장

**요청**:
```bash
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/app/output/document.md"
  }'
```

**요청 Body**:
```json
{
  "file_path": "/app/output/document.md"
}
```

**응답 예시**:
```json
{
  "message": "Successfully indexed document.md",
  "chunks": 15,
  "file_path": "/app/output/document.md"
}
```

---

### 2. POST /index-folder - 폴더 전체 인덱싱

**목적**: output 폴더의 모든 Markdown 파일을 일괄 인덱싱합니다.

**요청**:
```bash
curl -X POST "http://localhost:8000/index-folder" \
  -H "Content-Type: application/json" \
  -d '{
    "folder_path": "/app/output"
  }'
```

**요청 Body**:
```json
{
  "folder_path": "/app/output"
}
```

**응답 예시**:
```json
{
  "message": "Successfully indexed folder /app/output",
  "total_files": 50,
  "total_chunks": 750,
  "files": [
    {
      "file_path": "/app/output/doc1.md",
      "chunks": 15
    },
    {
      "file_path": "/app/output/doc2.md",
      "chunks": 20
    }
  ]
}
```

---

### 3. GET /documents - 인덱싱된 문서 목록

**목적**: ChromaDB에 저장된 모든 문서 목록을 조회합니다.

**요청**:
```bash
curl "http://localhost:8000/documents"
```

**응답 예시**:
```json
{
  "total_documents": 1322,
  "documents": [
    {
      "id": "doc_001_chunk_0",
      "source": "/app/output/document1.md",
      "chunk_index": 0
    },
    {
      "id": "doc_001_chunk_1",
      "source": "/app/output/document1.md",
      "chunk_index": 1
    }
  ]
}
```

---

## 🤖 RAG API (검색 및 질의응답)

### 1. POST /query - RAG 질의응답

**목적**: 사용자 질문에 대해 관련 문서를 검색하고 AI 답변을 생성합니다.

**주요 기능**:
- 질문 임베딩 생성
- ChromaDB 유사도 검색 (top_k=5)
- Ollama gemma2 모델로 답변 생성
- 출처 문서 제공

**요청**:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "5G 네트워크의 주요 특징은 무엇인가요?",
    "top_k": 5
  }'
```

**요청 Body**:
```json
{
  "question": "5G 네트워크의 주요 특징은 무엇인가요?",
  "top_k": 5
}
```

**응답 예시**:
```json
{
  "answer": "5G 네트워크의 주요 특징은 다음과 같습니다:\n\n1. **초고속 데이터 전송**: 최대 20Gbps의 다운로드 속도\n2. **초저지연**: 1ms 이하의 응답 시간\n3. **대규모 연결**: 1km² 당 100만 개 기기 동시 연결\n4. **네트워크 슬라이싱**: 용도별 가상 네트워크 구성\n\n이러한 특징들은 문서 5G.md에 상세히 설명되어 있습니다.",
  "sources": [
    {
      "content": "5G 네트워크는 초고속, 초저지연, 초연결을 특징으로 합니다...",
      "metadata": {
        "source": "/app/output/5G.md",
        "chunk_index": 0
      },
      "similarity": 0.92
    },
    {
      "content": "네트워크 슬라이싱 기술을 통해...",
      "metadata": {
        "source": "/app/output/5G.md",
        "chunk_index": 3
      },
      "similarity": 0.87
    }
  ],
  "question": "5G 네트워크의 주요 특징은 무엇인가요?"
}
```

**사용 시나리오**:
```bash
# Python 예시
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "question": "VoLTE와 ViLTE의 차이점은?",
        "top_k": 3
    }
)

result = response.json()
print(f"답변: {result['answer']}")
print(f"\n출처:")
for source in result['sources']:
    print(f"  - {source['metadata']['source']} (유사도: {source['similarity']})")
```

---

### 2. GET /search - 유사 문서 검색

**목적**: 검색어와 유사한 문서 조각들을 찾습니다 (AI 답변 없이).

**요청**:
```bash
curl "http://localhost:8000/search?query=5G%20네트워크&top_k=5"
```

**파라미터**:
- `query` (required): 검색어
- `top_k` (optional): 검색 결과 개수 (기본값: 5)

**응답 예시**:
```json
{
  "results": [
    {
      "content": "5G 네트워크는 초고속, 초저지연, 초연결을 특징으로 합니다...",
      "metadata": {
        "source": "/app/output/5G.md",
        "chunk_index": 0
      },
      "distance": 0.08
    },
    {
      "content": "네트워크 슬라이싱 기술을 통해...",
      "metadata": {
        "source": "/app/output/5G.md",
        "chunk_index": 3
      },
      "distance": 0.13
    },
    {
      "content": "femtocell은 5G 소형 기지국...",
      "metadata": {
        "source": "/app/output/S2-2311030_5G_femto_v3.md",
        "chunk_index": 2
      },
      "distance": 0.21
    }
  ],
  "total": 3
}
```

**사용 시나리오**:
- 특정 주제 관련 문서 찾기
- 답변 생성 전 관련 자료 확인
- 문서 연관성 분석

---

## 🛠️ 유틸리티 API

### 1. GET /health - 헬스 체크

**목적**: 서비스 상태를 확인합니다.

**요청**:
```bash
curl "http://localhost:8000/health"
```

**응답**:
```json
{
  "status": "healthy"
}
```

---

### 2. GET /supported-formats - 지원 파일 형식

**목적**: 변환 가능한 모든 파일 형식을 조회합니다.

**요청**:
```bash
curl "http://localhost:8000/supported-formats"
```

**응답 예시**:
```json
{
  "formats": [
    ".csv", ".doc", ".docx", ".epub", ".gif", ".htm", ".html",
    ".jpg", ".jpeg", ".json", ".md", ".mp3", ".m4a", ".pdf",
    ".png", ".pptx", ".ppt", ".tiff", ".txt", ".wav", ".xls",
    ".xlsx", ".xml", ".zip"
  ],
  "count": 26
}
```

---

## 🔗 통합 워크플로우 예시

### 시나리오 1: 변환 + 자동 인덱싱 + 질의응답

```bash
# 1. 파일 변환 (자동 인덱싱)
curl -X POST "http://localhost:8000/convert" \
  -F "file=@5G_whitepaper.pdf" \
  -F "auto_index=true"

# 2. 바로 질문하기
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "5G의 주파수 대역은?",
    "top_k": 3
  }'
```

---

### 시나리오 2: 대량 파일 배치 처리

```bash
# 1. 파일 복사 (예: 500개)
cp /archive/*.pdf ./input/

# 2. 배치 처리 시작
BATCH_ID=$(curl -X POST "http://localhost:8000/batch/folder" \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "/app/input", "batch_size": 100, "auto_index": true}' \
  | jq -r '.batch_id')

echo "Batch ID: $BATCH_ID"

# 3. 진행률 모니터링 (5초마다)
watch -n 5 "curl -s http://localhost:8000/batch/$BATCH_ID | jq '.progress_percentage'"

# 4. 완료 후 문서 목록 확인
curl "http://localhost:8000/documents" | jq '.total_documents'

# 5. 질의응답 테스트
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "전체 문서에서 VoLTE 관련 내용을 요약해줘"}'
```

---

### 시나리오 3: 기존 마크다운 인덱싱

```bash
# 1. output 폴더에 기존 .md 파일들이 있는 경우
ls ./output/*.md

# 2. 전체 폴더 인덱싱
curl -X POST "http://localhost:8000/index-folder" \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "/app/output"}'

# 3. 인덱싱 확인
curl "http://localhost:8000/documents"

# 4. 검색 테스트
curl "http://localhost:8000/search?query=femtocell&top_k=5"
```

---

## 📊 디렉토리 구조

```
markitdown/
├── input/                  # 입력 파일 폴더
│   ├── document1.pdf
│   ├── document2.docx
│   └── ...
├── output/                 # 변환된 Markdown 파일
│   ├── document1.md
│   ├── document2.md
│   └── ...
├── vector_store/           # ChromaDB 벡터 저장소
│   └── chroma.sqlite3
├── batch_state/            # 배치 작업 상태 (JSON)
│   ├── batch-20260207-143052-abc123.json
│   └── batch-20260207-150030-def456.json
├── app/
│   ├── converter.py        # FastAPI 애플리케이션
│   ├── batch_manager.py    # 배치 상태 관리
│   ├── indexer.py          # 문서 인덱싱
│   ├── rag.py              # RAG 파이프라인
│   └── ...
├── docker-compose.yml      # Docker Compose 설정
├── Dockerfile
└── requirements.txt
```

---

## 🔧 설정 및 환경 변수

### docker-compose.yml 주요 설정

```yaml
services:
  app:
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - LLM_MODEL=gemma2
      - EMBEDDING_MODEL=mxbai-embed-large
      - CHUNK_SIZE=512
      - CHUNK_OVERLAP=128
      - DEFAULT_BATCH_SIZE=100
    volumes:
      - ./input:/app/input
      - ./output:/app/output
      - ./vector_store:/app/vector_store
      - ./batch_state:/app/batch_state
```

### 주요 설정값

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `CHUNK_SIZE` | 512 | 청크당 문자 수 |
| `CHUNK_OVERLAP` | 128 | 청크 오버랩 |
| `DEFAULT_BATCH_SIZE` | 100 | 배치당 파일 수 |
| `TOP_K` | 5 | 검색 결과 개수 |
| `TEMPERATURE` | 0.7 | LLM 온도 |

---

## 🐛 문제 해결

### 1. Ollama 연결 실패

```bash
# Ollama 컨테이너 확인
docker compose ps

# Ollama 서버 상태
curl http://localhost:11434/api/tags

# 로그 확인
docker compose logs ollama
```

### 2. ChromaDB 오류

```bash
# 벡터 저장소 초기화
rm -rf ./vector_store/*
docker compose restart app

# 재인덱싱
curl -X POST "http://localhost:8000/index-folder" \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "/app/output"}'
```

### 3. 배치 처리 중단

```bash
# 배치 상태 확인
curl "http://localhost:8000/batch/{batch_id}"

# 재시작 (자동으로 중단 지점부터 재개)
curl -X POST "http://localhost:8000/batch/folder" \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "/app/input", "batch_size": 100}'
```

---

## 📖 API 문서

서버 실행 후 다음 주소에서 인터랙티브 문서를 확인할 수 있습니다:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🐳 Docker 명령어

```bash
# 서비스 시작
docker compose up -d --build

# 로그 확인
docker compose logs -f app

# 서비스 중지
docker compose down

# 전체 초기화 (볼륨 포함)
docker compose down -v
rm -rf ./vector_store/* ./batch_state/*

# 컨테이너 재시작
docker compose restart app
```

---

## 📚 추가 문서

- [아키텍쳐.md](./아키텍쳐.md) - 전체 시스템 설계
- [테스트 방법.md](./테스트%20방법.md) - 테스트 가이드
