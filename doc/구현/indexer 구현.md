# indexer 구현

## 개요

지정된 디렉토리(output)의 마크다운 파일을 벡터 저장소(Vector Store)에 자동으로 인덱싱하는 모듈입니다. 인덱싱 상태(Index State)를 디스크에 저장 및 관리하여, 변경된 파일만 효율적으로 재인덱싱합니다.

## 파일 경로

```
markitdown/app/indexer.py
```

## 주요 클래스

### 1. IndexStateManager

인덱싱 상태를 추적하고 파일 기반(JSON)으로 관리하는 클래스입니다.

- **상태 파일**: config에 정의된 JSON 파일(기본: `vector_store/index_state.json`)에 상태를 저장합니다.
- **주요 메서드**:
  - `is_indexed(file_path)`: 파일이 이미 인덱싱되어 있는지 확인합니다.
  - `needs_reindex(file_path)`: 파일 내용 변경 여부를 SHA256 해시 비교로 판단합니다.
  - `add_file(...)`: 인덱싱 완료된 파일 정보를 상태에 등록합니다.
  - `remove_file(...)`: 파일을 상태에서 제거하고 연결된 청크 ID 목록을 반환합니다.
- **상태 정보**: 파일 경로, 파일명, 해시값, 청크 ID 목록, 인덱싱 시간, 상태 등을 저장합니다.

### 2. DocumentIndexer

실제 문서 인덱싱 작업을 수행하는 클래스입니다.

- **구성 요소**: `VectorStore`, `DocumentEmbedder`, `IndexStateManager`를 조합하여 동작합니다.
- **`index_file` 메서드**:
  1. **검증**: 파일 존재 및 지원 포맷 여부 확인.
  2. **변경 감지**: `needs_reindex`를 통해 불필요한 작업 방지 (`force_reindex` 옵션 지원).
  3. **청소**: 재인덱싱 시 기존 청크를 벡터 저장소에서 삭제.
  4. **임베딩**: `DocumentEmbedder`를 사용해 문서를 청크로 분할하고 임베딩 생성.
  5. **저장**: 생성된 벡터와 메타데이터를 `VectorStore`에 추가.
  6. **상태 업데이트**: 성공 시 `IndexStateManager`에 기록.

## 데이터 흐름

1. 파일 시스템에서 `.md` 파일 감지.
2. `IndexStateManager`로 해시 비교 (변경 없으면 스킵).
3. `DocumentEmbedder`로 텍스트 -> [DocumentChunk].
4. `VectorStore`에 벡터 저장.
5. index_state.json 업데이트.
