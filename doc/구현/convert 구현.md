# Converter 구현 문서

## 개요

MarkItDown 라이브러리를 활용한 파일 변환 REST API 서비스입니다. FastAPI를 사용하여 다양한 형식의 파일을 Markdown으로 변환하고, Docker 컨테이너로 배포할 수 있습니다.

## 주요 기능

### 1. 단일 파일 변환 (`/convert`)
- 파일 업로드 후 Markdown으로 변환
- 변환된 파일을 직접 다운로드
- output 폴더에도 자동 저장

### 2. 폴더 배치 변환 (`/convert-folder`)
- input 폴더 내 모든 지원 파일을 순차 변환
- 변환 결과를 JSON으로 다운로드
- 각 파일별 성공/실패 여부 및 소요 시간 기록

### 3. 지원 포맷 조회 (`/supported-formats`)
- 지원하는 파일 형식 목록 반환

### 4. 헬스 체크 (`/health`)
- 서비스 상태 확인

## 지원 파일 형식

총 26가지 파일 형식 지원:

| 카테고리 | 확장자 |
|---------|--------|
| **문서** | `.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt`, `.xlsx`, `.xls` |
| **텍스트** | `.txt`, `.csv`, `.json`, `.xml`, `.html`, `.htm`, `.md` |
| **이미지** | `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff` |
| **미디어** | `.wav`, `.mp3`, `.m4a`, `.flac` |
| **압축** | `.zip`, `.epub` |

## 아키텍처

### 디렉토리 구조

```
markitdown/
├── app/
│   └── converter.py          # 메인 애플리케이션
├── input/                    # 배치 변환 입력 폴더
├── output/                   # 변환 결과 저장 폴더
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

### 주요 컴포넌트

#### 1. Config 클래스
설정값 중앙 관리:
- `OUTPUT_DIR`: 변환 파일 저장 경로 (`/app/output`)
- `INPUT_DIR`: 배치 입력 폴더 (`/app/input`)
- `SUPPORTED_FORMATS`: 지원 파일 형식 목록
- `LIBREOFFICE_TIMEOUT`: LibreOffice 변환 타임아웃 (60초)

#### 2. 변환 함수

**convert_doc_to_docx()**
- `.doc` → `.docx` 변환 (LibreOffice 사용)
- 타임아웃: 60초
- 반환: `(docx_path, error_message)`

**extract_markdown()**
- MarkItDown으로 Markdown 추출
- `text_content` 속성 사용
- Form feed 문자 제거

**save_markdown()**
- Markdown 텍스트를 파일로 저장
- UTF-8 인코딩 사용

**convert_single_file()**
- 단일 파일 변환 전체 프로세스
- 로그 출력 옵션
- 반환: `(success, filename, message, duration)`

#### 3. 로깅 시스템

실시간 변환 진행 상황을 터미널에 출력:

```
📥 Received file: document.pdf
🔄 Converting: document.pdf
✅ Success: document.pdf → document.md (1.23s)
```

배치 변환 시:

```
📂 Starting batch conversion: 5 files found
============================================================

[1/5]
🔄 Converting: report.docx
✅ Success: report.docx → report.md (0.87s)

[2/5]
🔄 Converting: old.doc
  📄 Converting .doc to .docx...
✅ Success: old.doc → old.md (1.45s)

============================================================
🏁 Batch conversion complete
   Total: 5 files
   ✅ Success: 4
   ❌ Failed: 1
   ⏱️  Total time: 5.32s
============================================================
```

## Docker 구성

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# LibreOffice 설치 (.doc 변환용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN mkdir -p /app/output /app/input

VOLUME ["/app/output", "/app/input"]

EXPOSE 8000

CMD ["uvicorn", "app.converter:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  markitdown-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./output:/app/output
      - ./input:/app/input
    environment:
      - MARKITDOWN_OUTPUT_DIR=/app/output
      - MARKITDOWN_INPUT_DIR=/app/input
    restart: unless-stopped
```

### 의존성 (requirements.txt)

```
fastapi==0.115.8
markitdown==0.0.1
python-multipart==0.0.9
uvicorn[standard]==0.30.6
```

## API 엔드포인트 상세

### POST /convert

**설명**: 단일 파일을 Markdown으로 변환

**요청**:
```bash
curl -X POST "http://localhost:8000/convert" \
  -F "file=@document.pdf" \
  -o result.md
```

**응답**: Markdown 파일 (직접 다운로드)

**처리 과정**:
1. 파일 업로드 수신
2. 임시 파일로 저장
3. 형식 확인
4. `.doc` 파일인 경우 `.docx`로 변환
5. MarkItDown으로 Markdown 추출
6. output 폴더에 저장
7. 파일 다운로드 응답
8. 임시 파일 정리

### POST /convert-folder

**설명**: input 폴더의 모든 지원 파일을 배치 변환

**요청**:
```bash
# 1. input 폴더에 파일 복사
cp *.pdf *.docx ./input/

# 2. 배치 변환 실행
curl -X POST "http://localhost:8000/convert-folder" \
  -o conversion_result.json
```

**응답**: JSON 파일

```json
{
  "total_files": 5,
  "converted_files": 4,
  "failed_files": 1,
  "total_duration": 5.32,
  "files": [
    {
      "input": "document.pdf",
      "output": "document.md",
      "status": "success",
      "duration": 1.23
    },
    {
      "input": "corrupted.pdf",
      "status": "failed",
      "reason": "Error: Failed to extract markdown",
      "duration": 0.15
    }
  ],
  "message": "Batch conversion complete: 4 succeeded, 1 failed"
}
```

**처리 과정**:
1. input 폴더에서 지원 파일 탐색
2. 각 파일을 순차적으로 변환
3. 진행 상황을 터미널에 로깅
4. 성공/실패 결과 수집
5. JSON 파일로 저장
6. JSON 다운로드 응답

### GET /supported-formats

**응답**:
```json
{
  "formats": [
    ".bmp", ".csv", ".doc", ".docx", ".epub",
    ".flac", ".gif", ".htm", ".html", ".jpeg",
    ".jpg", ".json", ".m4a", ".md", ".mp3",
    ".pdf", ".png", ".ppt", ".pptx", ".tiff",
    ".txt", ".wav", ".xls", ".xlsx", ".xml", ".zip"
  ],
  "count": 26
}
```

### GET /health

**응답**:
```json
{
  "status": "healthy"
}
```

## .doc 파일 처리

### 문제점
- Pandoc은 `.docx`만 지원, `.doc` 미지원
- MarkItDown도 `.doc` 직접 변환 불가

### 해결 방법
LibreOffice Headless 모드 사용:

```python
def convert_doc_to_docx(doc_path: Path) -> tuple[Optional[Path], str]:
    result = subprocess.run(
        [
            "libreoffice",
            "--headless",
            "--convert-to", "docx",
            "--outdir", str(doc_path.parent),
            str(doc_path)
        ],
        capture_output=True,
        timeout=60,
        text=True
    )
    
    docx_path = doc_path.parent / f"{doc_path.stem}.docx"
    
    if result.returncode == 0 and docx_path.exists():
        return docx_path, ""
    else:
        return None, "LibreOffice conversion failed"
```

### 변환 시간
- 50페이지 `.doc` 파일: 약 0.2초
- 전체 프로세스 (`.doc` → `.docx` → Markdown): 1~3초

## 코드 리팩토링 구조

### 주요 개선사항

1. **설정 분리**: `Config` 클래스로 중앙화
2. **단일 책임 원칙**: 함수별 명확한 역할 분리
3. **헬퍼 함수**: 재사용 가능한 유틸리티
4. **에러 처리**: 각 단계별 상세 에러 메시지
5. **문서화**: 모든 함수에 docstring 추가
6. **타입 힌팅**: 완벽한 타입 어노테이션

### 함수 구조

```
Configuration
├── Config 클래스
└── setup_logging()

Helper Functions
├── is_supported_format()
├── get_supported_files()
└── cleanup_temp_file()

Conversion Functions
├── convert_doc_to_docx()
├── extract_markdown()
├── save_markdown()
└── convert_single_file()

Result Handling
├── save_result_json()
├── log_batch_summary()
└── create_batch_result()

API Endpoints
├── convert_file()
├── convert_folder()
├── health_check()
└── get_supported_formats()
```

## 배포 및 실행

### 빌드 및 실행

```bash
# 빌드 및 실행
docker-compose up --build

# 백그라운드 실행
docker-compose up -d --build

# 재시작
docker-compose restart

# 종료
docker-compose down
```

### 로컬 개발

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 개발 서버 실행
uvicorn app.converter:app --reload
```

## 사용 예시

### 1. 단일 PDF 변환

```bash
curl -X POST "http://localhost:8000/convert" \
  -F "file=@report.pdf" \
  -o report.md
```

### 2. 구형 Word 문서 변환

```bash
curl -X POST "http://localhost:8000/convert" \
  -F "file=@legacy.doc" \
  -o legacy.md
```

### 3. 폴더 배치 변환

```bash
# 파일 준비
cp documents/*.{pdf,docx,xlsx} ./input/

# 변환 실행
curl -X POST "http://localhost:8000/convert-folder" \
  -o result.json

# 결과 확인
cat result.json | jq '.'

# 변환된 파일 확인
ls ./output/
```

## 성능 특성

### 단일 파일 변환 시간 (참고)

| 파일 형식 | 페이지/크기 | 예상 시간 |
|-----------|------------|----------|
| PDF | 10페이지 | 0.5~1.5초 |
| DOCX | 20페이지 | 0.8~2초 |
| DOC | 50페이지 | 1.3~2.3초 |
| XLSX | 5시트 | 0.3~0.8초 |
| PNG (OCR) | 1MB | 1~3초 |

### 배치 변환
- 파일당 평균 1~2초
- 100개 파일: 약 2~3분
- 순차 처리 (병렬 처리 미구현)

## 제한사항

1. **이미지 추출 미지원**
   - PDF, DOCX, XLSX 내 이미지는 추출되지 않음
   - 텍스트만 Markdown으로 변환

2. **순차 처리**
   - 배치 변환 시 파일을 순차적으로 처리
   - 대량 파일 처리 시 시간 소요

3. **메모리 사용**
   - 대용량 파일 처리 시 메모리 사용량 증가
   - LibreOffice 프로세스별 메모리 필요

## 향후 개선 가능 사항

1. **이미지 처리**
   - PDF/DOCX/XLSX 이미지 추출 및 저장
   - Markdown에 이미지 링크 포함

2. **병렬 처리**
   - 배치 변환 시 멀티프로세싱 적용
   - 처리 속도 대폭 개선

3. **진행률 API**
   - WebSocket으로 실시간 진행률 전송
   - 프론트엔드에서 진행 상황 표시

4. **변환 옵션**
   - OCR 옵션 설정
   - 테이블 포맷 선택
   - 이미지 품질 조정

5. **캐싱**
   - 동일 파일 재변환 시 캐시 사용
   - 변환 시간 단축

## API 문서

실행 후 자동 생성되는 문서:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 트러블슈팅

### 1. .doc 변환 실패
**증상**: "LibreOffice conversion failed" 에러

**해결**:
- LibreOffice가 설치되었는지 확인
- Docker 이미지 재빌드: `docker-compose up --build`

### 2. 변환 타임아웃
**증상**: 60초 이상 소요되는 대용량 파일

**해결**:
- `Config.LIBREOFFICE_TIMEOUT` 값 증가
- 파일 크기 줄이기

### 3. 메모리 부족
**증상**: 대량 파일 배치 변환 시 컨테이너 종료

**해결**:
- Docker 메모리 제한 증가
- 파일을 여러 배치로 나누어 처리

### 4. 로그가 보이지 않음
**증상**: 터미널에 변환 진행 상황 미표시

**해결**:
```bash
# 로그 확인
docker-compose logs -f markitdown-api
```

## 참고 자료

- [MarkItDown GitHub](https://github.com/microsoft/markitdown)
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [LibreOffice Headless](https://help.libreoffice.org/latest/en-US/text/shared/guide/headless.html)
