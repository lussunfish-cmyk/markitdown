# convert 구현

## 개요

문서 변환, 인덱싱, RAG(검색 증강 생성), 평가 기능을 통합 제공하는 REST API 서비스입니다. FastAPI 애플리케이션의 엔트리포인트로서, 파일 변환부터 벡터 DB 인덱싱, 질의응답까지의 전체 파이프라인을 관리합니다.

## 파일 경로

```
markitdown/app/converter.py
```

## 아키텍처 및 주요 기능

### 1. FastAPI 애플리케이션

- `FastAPI` 인스턴스 생성 및 설정 (Title, Version, Description from Config).
- 정적 파일(Static Files) 마운트 (`/static`).
- 로깅 설정 (`setup_logging`).
- `batch_manager`, `indexer`, `rag` 등 하위 모듈과 연동.

### 2. 파일 변환 (Conversion)

#### 주요 함수
- `convert_single_file`: 단일 파일을 변환하는 핵심 로직.
  - 지원 포맷 확인.
  - `.doc` 파일의 경우 `LibreOffice`를 사용하여 `.docx`로 우선 변환.
  - `MarkItDown`을 사용하여 Markdown 텍스트 추출.
  - 추출된 텍스트를 `.md` 파일로 저장.
  - 임시 파일 정리.
- `extract_markdown`: MarkItDown 라이브러리 래퍼.
- `convert_doc_to_docx`: `subprocess`를 통해 LibreOffice CLI 실행.

#### LibreOffice 변환
- MarkItDown이 `.doc` (구형 바이너리 포맷)를 직접 지원하지 않을 수 있어, LibreOffice를 중간 변환기로 사용합니다.
- 네이티브 실행 환경에서 `libreoffice` 설치가 필요합니다.

### 3. API 엔드포인트 (Endpoints)

#### `/convert` (POST)
- 단일 파일 업로드 및 변환.
- 변환된 파일을 즉시 다운로드(FileResponse)하거나 JSON 결과를 반환할 수 있습니다.
- `auto_index` 옵션을 통해 변환 후 즉시 벡터 DB에 인덱싱 가능.

#### `/convert-folder` (POST)
- 서버 내부의 `input` 폴더에 있는 모든 파일을 일괄 변환.
- 변환 진행률 및 결과를 JSON으로 반환.
- 대량 처리에 적합.
- `auto_index` 옵션 지원.

#### `/convert-batch` (POST)
- 여러 파일을 한 번에 업로드하여 배치 작업으로 처리.
- `BatchManager`를 통해 상태를 관리하며 비동기적으로 처리.

#### `/batch/folder` (POST)
- 서버 내부 폴더의 대량 파일을 배치 작업으로 등록.
- 파일 업로드 없이 대규모 데이터셋 처리에 최적화.

#### `/batch/{batch_id}` (GET, DELETE)
- 배치 작업의 상태 조회 및 삭제.

#### `/index`, `/index-folder` (POST)
- 변환된 마크다운 파일(단일 또는 폴더)을 벡터 DB에 인덱싱.
- `force` 옵션으로 재인덱싱 지원.

#### `/query` (POST)
- RAG 질의응답 엔드포인트.
- 질문을 받아 관련 문서를 검색하고 LLM을 통해 답변 생성.
- 답변과 함께 근거 문서(Sources) 반환.

#### `/supported-formats` (GET)
- 지원하는 파일 확장자 목록(config 기준)을 반환합니다.

#### `/search` (GET)
- 문서 검색 전용 엔드포인트.
- LLM 답변 생성 없이, 벡터 DB에서 검색된 문서 청크만 반환.
- 디버깅 및 검색 품질 테스트 용도.

#### `/testset/generate` (POST)
- Ragas를 사용하여 문서 기반의 테스트셋(질문-답변 쌍) 자동 생성.

#### `/evaluate` (POST)
- 생성된 테스트셋을 사용하여 검색 성능(Recall, Precision 등) 평가.

#### `/health` (GET)
- 서비스 상태 확인 (Ollama 연결, Vector Store 상태 등).

### 4. 결과 관리
- 변환 결과는 `output` 디렉토리에 `.md` 파일로 저장됩니다.
- 변환 메타데이터(파일명, 성공 여부, 소요 시간 등)는 JSON으로 기록됩니다.
- 배치 작업 상태는 `batch_state` 디렉토리에 JSON으로 영구 저장됩니다.

## 데이터 흐름

1. 클라이언트 `POST /convert` 요청 (파일 업로드)
2. `converter.py` 가 파일 형식 확인
3. `.doc`인 경우 `LibreOffice`로 `.docx` 변환
4. `MarkItDown` 라이브러리로 텍스트 추출
5. `output/` 폴더에 `.md` 저장
6. (옵션) `DocumentIndexer`를 통해 벡터 DB 인덱싱
7. JSON 결과 반환
