# indexer.py 구현

## 개요

output 폴더의 마크다운 파일을 자동으로 벡터 스토어에 인덱싱하는 모듈입니다. 
인덱싱 상태를 관리하여 변경된 파일만 재인덱싱하고, 배치 처리 및 개별 파일 처리를 모두 지원합니다.

## 파일 경로

```
markitdown/app/indexer.py (약 1000줄)
markitdown/test_indexer.py (테스트 파일)
```

## 왜 필요한가?

### 문제점

1. **반복적인 코드**: DocumentEmbedder + VectorStore 사용할 때마다 같은 로직 반복
2. **상태 추적 부재**: 어떤 파일이 인덱싱되었는지 알 수 없음
3. **중복 인덱싱**: 같은 파일을 여러 번 인덱싱할 수 있음
4. **변경 감지 없음**: 파일이 수정되었는지 알 수 없음
5. **배치 처리 불편**: 여러 파일 처리 시 에러 처리 복잡

### 해결책

indexer.py는 다음을 제공합니다:

```python
# 간단한 사용
indexer = get_indexer()

# 단일 파일
indexer.index_file(Path("output/document.md"))

# 전체 디렉토리
indexer.index_directory()

# 상태 조회
docs = indexer.get_indexed_files()
stats = indexer.get_stats()

# 삭제
indexer.delete_document(Path("output/document.md"))
```

## 주요 구성 요소

### 1. IndexStateManager

인덱싱 상태를 추적하고 관리하는 클래스입니다.

#### 주요 메서드

```python
# 파일 인덱싱 상태 확인
state_manager.is_indexed(file_path)        # 인덱싱됨?
state_manager.needs_reindex(file_path)     # 재인덱싱 필요?

# 파일 정보 추가/제거
state_manager.add_file(file_path, chunk_ids, num_chunks)
state_manager.remove_file(file_path)

# 상태 조회
state_manager.get_indexed_files()          # 모든 인덱싱 파일
state_manager.get_file_info(file_path)     # 특정 파일 정보

# 상태 저장/로드
state_manager.save_state()                 # 디스크 저장
state_manager.load_state()                 # 디스크 로드

# 초기화
state_manager.clear()                      # 전체 상태 초기화
```

#### 내부 동작

- **파일 해시**: SHA256 해시로 파일 변경 감지
  - 파일이 수정되면 해시가 바뀜
  - 해시가 다르면 재인덱싱 필요 표시

- **상태 저장**: JSON 파일로 영구 저장
  - 경로: `/app/vector_store/index_state.json`
  - 형식: 파일 경로를 키로 사용

- **상태 구조**:
  ```json
  {
    "files": {
      "/app/output/document.md": {
        "path": "/app/output/document.md",
        "filename": "document.md",
        "hash": "abc123...",
        "chunk_ids": ["chunk_0", "chunk_1", ...],
        "num_chunks": 10,
        "status": "indexed",
        "indexed_at": "2026-02-07T08:06:05.468424"
      }
    }
  }
  ```

### 2. DocumentIndexer

문서를 벡터 스토어에 인덱싱하는 메인 클래스입니다.

#### 주요 메서드

##### index_file()
```python
result = indexer.index_file(
    file_path=Path("output/document.md"),
    force_reindex=False  # 강제 재인덱싱 여부
)

# 반환값
{
    "filename": "document.md",
    "path": "/app/output/document.md",
    "status": "indexed" | "skipped" | "failed",
    "chunks": 10,
    "message": "성공 메시지 또는 에러 메시지"
}
```

**동작 흐름:**
1. 파일 존재 여부 확인
2. 지원 형식 확인 (.md, .txt)
3. 재인덱싱 필요 여부 확인 (해시 기반)
4. 기존 청크 삭제 (재인덱싱 시)
5. 문서 임베딩
6. 벡터 스토어에 추가
7. 상태 저장

##### index_directory()
```python
result = indexer.index_directory(
    directory=None,          # None이면 기본 output 디렉토리
    force_reindex=False,     # 강제 재인덱싱
    recursive=False          # 하위 디렉토리 포함
)

# 반환값: IndexResponse
{
    "total_files": 6,           # 발견된 파일 수
    "indexed_files": 4,         # 성공 인덱싱
    "failed_files": 1,          # 실패
    "total_chunks": 624,        # 총 청크 수
    "files": [
        {
            "filename": "document.md",
            "status": "indexed",
            "chunks": 40,
            ...
        },
        ...
    ]
}
```

**특징:**
- 모든 .md, .txt 파일 자동 발견
- 병렬 처리하지 않음 (순차 처리)
- 중간 실패해도 계속 진행
- 진행 상황 로깅

##### rebuild_index()
```python
result = indexer.rebuild_index()
```

- 전체 인덱스 초기화
- 모든 파일 재인덱싱
- 주의: 데이터 손실 가능

##### delete_document()
```python
success = indexer.delete_document(Path("output/document.md"))
```

- 문서 삭제
- 상태에서 제거
- 벡터 스토어에서 청크 삭제

##### delete_by_source()
```python
success = indexer.delete_by_source("document.md")
```

- 소스 파일명으로 삭제

##### get_indexed_files()
```python
documents = indexer.get_indexed_files()

# 반환값: List[Document]
[
    {
        "id": "document",
        "filename": "document.md",
        "total_chunks": 10,
        "indexed_at": "2026-02-07T08:06:05...",
        "status": "indexed"
    },
    ...
]
```

##### get_stats()
```python
stats = indexer.get_stats()

# 반환값
{
    "total_files": 7,
    "total_chunks": 627,
    "vector_store_count": 624,
    "status": {"indexed": 7},
    "files": ["document1.md", "document2.md", ...]
}
```

## 팩토리 함수

### get_indexer()
```python
# 싱글톤 인스턴스 반환 (애플리케이션 전체에서 동일 인스턴스)
indexer = get_indexer()
```

### create_indexer()
```python
# 새로운 인스턴스 생성
indexer = create_indexer(
    vector_store=None,      # 기본값 사용
    embedder=None,          # 기본값 사용
    document_dir=None       # 기본값: /app/output
)
```

## 사용 예시

### 예시 1: 기본 사용

```python
from app.indexer import get_indexer

# 인덱서 가져오기
indexer = get_indexer()

# 전체 디렉토리 인덱싱
result = indexer.index_directory()

print(f"인덱싱 결과:")
print(f"  총 파일: {result.total_files}")
print(f"  성공: {result.indexed_files}")
print(f"  실패: {result.failed_files}")
print(f"  총 청크: {result.total_chunks}")
```

### 예시 2: 단일 파일 인덱싱

```python
from pathlib import Path
from app.indexer import get_indexer

indexer = get_indexer()

# 특정 파일 인덱싱
result = indexer.index_file(Path("/app/output/my_doc.md"))

if result["status"] == "indexed":
    print(f"✓ {result['chunks']}개 청크 생성")
elif result["status"] == "skipped":
    print(f"⊘ 스킵됨: {result['message']}")
else:
    print(f"✗ 실패: {result['message']}")
```

### 예시 3: 강제 재인덱싱

```python
from pathlib import Path
from app.indexer import get_indexer

indexer = get_indexer()

# 파일 수정된 경우만 인덱싱
result = indexer.index_file(Path("/app/output/doc.md"))

# 무조건 재인덱싱 (이전 데이터 삭제)
result = indexer.index_file(
    Path("/app/output/doc.md"),
    force_reindex=True
)
```

### 예시 4: 문서 삭제

```python
from pathlib import Path
from app.indexer import get_indexer

indexer = get_indexer()

# 파일 기반 삭제
success = indexer.delete_document(Path("/app/output/old_doc.md"))

# 소스명 기반 삭제
success = indexer.delete_by_source("old_doc.md")
```

### 예시 5: 통계 조회

```python
from app.indexer import get_indexer

indexer = get_indexer()

stats = indexer.get_stats()

print(f"총 인덱싱된 파일: {stats['total_files']}")
print(f"총 청크: {stats['total_chunks']}")
print(f"벡터 스토어 항목: {stats['vector_store_count']}")
print(f"파일 목록: {stats['files']}")
```

## 테스트

### 테스트 파일

```
markitdown/test_indexer.py
```

### 테스트 실행

```bash
# 컨테이너 내에서 실행
docker compose exec -T markitdown-api python test_indexer.py

# 또는 로컬에서
python test_indexer.py
```

### 테스트 항목

1. **TEST 1**: IndexStateManager
   - 파일 추가/조회
   - 상태 저장/로드

2. **TEST 2**: 단일 파일 인덱싱
   - 새 파일 인덱싱
   - 중복 인덱싱 스킵
   - 강제 재인덱싱

3. **TEST 3**: 디렉토리 배치 인덱싱
   - 6개 마크다운 파일 인덱싱
   - 624개 청크 생성

4. **TEST 4**: 인덱싱된 파일 목록
   - Document 객체 조회

5. **TEST 5**: 인덱싱 통계
   - 전체 통계 조회

6. **TEST 6**: 문서 삭제
   - 문서 삭제 및 상태 확인

### 예상 테스트 결과

```
======================================================================
인덱싱 완료
  총 파일: 6
  인덱싱됨: 0 (이전에 인덱싱됨)
  실패: 0
  스킵됨: 6
  총 청크: 624
======================================================================

인덱싱된 문서 수: 7
1. test.md (3 청크)
2. 46001-400.md (24 청크)
3. 5G.md (4 청크)
...

인덱싱 통계:
  총 파일: 7
  총 청크: 627
  벡터 스토어 항목: 624
  상태별 집계: {'indexed': 7}

✓ 모든 테스트 완료!
```

## 주요 특징

### 1. 변경 감지

```python
# 시나리오 1: 파일이 변경되지 않음
result = indexer.index_file(path)  # status: "skipped"

# 시나리오 2: 파일이 수정됨
# 파일 내용 수정 후
result = indexer.index_file(path)  # status: "indexed"

# 시나리오 3: 강제 재인덱싱
result = indexer.index_file(path, force_reindex=True)  # status: "indexed"
```

### 2. 에러 처리

- 존재하지 않는 파일: `failed` 상태
- 지원하지 않는 형식: `failed` 상태
- 임베딩 생성 실패: `failed` 상태
- 부분 실패해도 전체 중단 안 함

### 3. 상태 영속성

- JSON 파일로 상태 저장
- 서버 재시작 후에도 상태 유지
- 외부에서 직접 수정 가능 (주의)

### 4. 성능 최적화

- 이미 인덱싱된 파일 스킵
- 변경되지 않은 파일 스킵
- 배치 추가로 벡터 스토어 부하 감소

## 주의사항

### 1. 파일 경로

```python
# ✓ 올바른 절대 경로
indexer.index_file(Path("/app/output/document.md"))

# ✓ 상대 경로 (권장)
indexer.index_file(Path("output/document.md"))

# ✗ 부정확한 경로
indexer.index_file("output/document.md")  # 문자열
```

### 2. 지원 형식

```python
# 지원 형식: .md, .txt
# config.INDEXING.SUPPORTED_FORMATS

# 다른 형식은 "failed" 상태로 반환
result = indexer.index_file(Path("document.pdf"))
# result["status"] == "failed"
```

### 3. 상태 관리

```python
# 상태 파일 위치
# /app/vector_store/index_state.json

# 상태 파일 초기화 (주의!)
state_manager.clear()  # 모든 인덱싱 정보 삭제

# 벡터 스토어와 상태 파일이 불일치할 수 있음
# → 재인덱싱으로 해결
indexer.rebuild_index()
```

### 4. 동시성

- **관현 구현**: 동시성 미지원 (순차 처리만)
- 멀티 스레드/프로세스 사용 시 상태 파일 충돌 가능
- 단일 스레드 사용 권장

### 5. 메모리 사용

- 큰 파일(수 MB): 메모리 사용량 증가
- 청킹 크기 조정으로 개선 가능
- config.CHUNKING.CHUNK_SIZE 수정

## 향후 개선사항

1. **멀티 스레드 지원**: 동시성 처리로 성능 개선
2. **진행률 표시**: tqdm 라이브러리 추가
3. **파일 감시**: watchdog으로 자동 인덱싱
4. **증분 인덱싱**: 논리적 변경만 감지
5. **메타데이터 확장**: 작성자, 태그 등 추가

## 참고 자료

- [config.py 구현](config%20구현.md) - 설정 옵션
- [embedding.py 구현](embedding%20구현.md) - 임베딩 생성
- [vector_store.py 구현](vector_store%20구현.md) - 벡터 저장소

## 결론

indexer.py는 RAG 시스템의 핵심 자동화 모듈입니다. 
상태 관리를 통해 효율적인 인덱싱을 지원하며, 
API 엔드포인트와 함께 완전한 문서 관리 시스템을 구성합니다.
