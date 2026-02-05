# MarkItDown REST API

FastAPI를 사용한 MarkItDown 변환 서비스입니다. 다양한 파일 형식을 Markdown으로 변환할 수 있습니다.

## 기능

- 파일 업로드 후 자동으로 Markdown으로 변환
- 변환된 파일을 호스트 머신의 볼륨에 자동 저장
- Docker Compose를 통한 간편한 배포

## 사전 요구사항

- Docker & Docker Compose
- 또는 Python 3.11+

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
- 포트 8000을 노출

## 사용 방법

### 파일 변환 (curl)

```bash
curl -X POST "http://localhost:8000/convert" \
  -F "file=@/path/to/input.pdf" \
  -o response.json

# 또는 다른 형식
curl -X POST "http://localhost:8000/convert" \
  -F "file=@/path/to/document.docx"
```

### 응답 예시

```json
{
  "filename": "document.md",
  "message": "File converted successfully and saved to /app/output/document.md"
}
```

변환된 파일은 자동으로 `./output` 디렉토리에 저장됩니다.

## 수동 실행 (Docker)

```bash
# 이미지 빌드
docker build -t markitdown-api .

# 컨테이너 실행 (볼륨 마운트 포함)
docker run --rm -p 8000:8000 -v $(pwd)/output:/app/output markitdown-api
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
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 지원 파일 형식

MarkItDown이 지원하는 모든 형식:
- PDF (.pdf)
- Office 문서 (.docx, .pptx, .xlsx)
- HTML (.html)
- 이미지 (.png, .jpg, 등 - OCR 지원)
- 텍스트 파일 (.txt, .md, 등)
- 그 외 여러 형식

## 환경 변수

- `MARKITDOWN_OUTPUT_DIR`: 변환된 파일 저장 경로 (기본값: `/app/output`)

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
