# MarkItDown REST API

FastAPI를 사용한 MarkItDown 변환 서비스입니다. 다양한 파일 형식을 Markdown으로 변환할 수 있습니다.

## 기능

- **단일 파일 변환**: 파일 업로드 후 자동으로 Markdown으로 변환
- **폴더 배치 변환**: 폴더 내 모든 지원 파일을 순차적으로 변환
- **변환된 파일 자동 저장**: 호스트 머신의 볼륨에 자동 저장
- **Docker Compose**: 간편한 배포

## 지원 파일 형식

- **문서**: PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS
- **텍스트**: TXT, CSV, JSON, XML, HTML, HTM, MD
- **이미지**: PNG, JPG, JPEG, GIF, BMP, TIFF
- **미디어**: WAV, MP3, M4A, FLAC
- **압축**: ZIP, EPUB

## 빠른 시작 (Docker Compose)

```bash
# 컨테이너 빌드 및 실행
docker-compose up --build

# 또는 백그라운드 실행
docker-compose up -d --build
```

이 명령어는 자동으로:
- 이미지를 빌드
- `./output` 디렉토리를 컨테이너의 `/app/output`에 마운트
- `./input` 디렉토리를 컨테이너의 `/app/input`에 마운트
- 포트 8000을 노출

## 사용 방법

### 1. 개별 파일 변환

```bash
curl -X POST "http://localhost:8000/convert" \
  -F "file=@/path/to/document.pdf"
```

**응답 예시**:
```json
{
  "filename": "document.md",
  "message": "Converted successfully"
}
```

### 2. 폴더 배치 변환

먼저 변환할 파일들을 `./input` 디렉토리에 복사합니다:

```bash
cp /path/to/files/* ./input/
```

그 다음 폴더 변환 API를 호출합니다:

```bash
curl -X POST "http://localhost:8000/convert-folder"
```

**응답 예시**:
```json
{
  "total_files": 5,
  "converted_files": 4,
  "failed_files": 1,
  "files": [
    {
      "input": "document1.pdf",
      "output": "document1.md",
      "status": "success"
    },
    {
      "input": "document2.docx",
      "output": "document2.md",
      "status": "success"
    },
    {
      "input": "image.jpg",
      "output": "image.md",
      "status": "success"
    },
    {
      "input": "unsupported.xyz",
      "status": "failed",
      "reason": "Unsupported format: .xyz"
    }
  ],
  "message": "Batch conversion complete: 4 succeeded, 1 failed"
}
```

### 3. 지원 파일 형식 확인

```bash
curl "http://localhost:8000/supported-formats"
```

**응답**:
```json
{
  "formats": [".csv", ".doc", ".docx", ".epub", ".gif", ...],
  "count": 26
}
```

### 4. 헬스 체크

```bash
curl "http://localhost:8000/health"
```

**응답**:
```json
{"status": "healthy"}
```

## 디렉토리 구조

```
markitdown/
├── input/              # 배치 변환용 입력 폴더
├── output/             # 변환된 md 파일 저장 폴더
├── app/
│   └── main.py        # FastAPI 애플리케이션
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── USAGE.md
```

## 수동 실행 (Docker)

```bash
# 이미지 빌드
docker build -t markitdown-api .

# 컨테이너 실행 (볼륨 마운트 포함)
docker run --rm -p 8000:8000 \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/input:/app/input \
  markitdown-api
```

## 로컬 개발

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn app.main:app --reload
```

## API 문서

서버 실행 후 다음 주소에서 인터랙티브 문서 확인 가능:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 환경 변수

- `MARKITDOWN_OUTPUT_DIR`: 변환된 파일 저장 경로 (기본값: `/app/output`)
- `MARKITDOWN_INPUT_DIR`: 배치 변환 입력 폴더 (기본값: `/app/input`)

## 컨테이너 종료

```bash
# Docker Compose 사용 시
docker-compose down

# 일반 Docker 사용 시
docker stop <container_id>
```

## 스토리지 위치

- **Docker Compose**: `./output` 디렉토리 (프로젝트 루트)
- **Docker 수동 실행**: `-v` 옵션에서 지정한 경로

## 폴더 배치 처리 예시

```bash
# input 폴더에 여러 파일 준비
ls ./input/
# document.pdf
# presentation.pptx
# spreadsheet.xlsx
# report.docx

# 배치 변환 실행
curl -X POST "http://localhost:8000/convert-folder"

# 결과 확인
ls ./output/
# document.md
# presentation.md
# spreadsheet.md
# report.md
```
