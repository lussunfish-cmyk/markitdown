# Vector Store 구현

## 개요

임베딩된 문서 벡터를 저장하고 검색하는 저장소(Vector Store) 구현입니다. 현재는 **ChromaDB**를 메인 백엔드로 사용하며, 추상화 레이어를 통해 향후 FAISS 등 다른 저장소로의 확장을 고려하여 설계되었습니다.

## 파일 경로

```
markitdown/app/vector_store.py
```

## 주요 클래스

### 1. VectorStore (Abstract Base Class)

모든 벡터 저장소가 구현해야 할 공통 인터페이스를 정의합니다.

- `add`: 문서 및 벡터 추가.
- `search`: 유사도 검색.
- `delete`: ID 기반 삭제.
- `get`: ID 기반 조회.
- `count`: 총 문서 수 반환.
- `clear`: 저장소 초기화.
- `get_collection_info`: 메타데이터 및 통계 정보.

### 2. ChromaVectorStore (Implementation)

**ChromaDB**를 사용한 실제 구현체입니다.

- **초기화**: `chromadb.PersistentClient`를 사용하여 로컬 파일 시스템에 데이터를 영구 저장합니다. 코사인 유사도(`hnsw:space="cosine"`)를 사용하도록 설정합니다.
- **데이터 구조**: `upsert`를 사용하여 중복 ID 발생 시 내용을 갱신합니다.
- **검색 (`search`)**:
  - 벡터 유사도 검색 수행.
  - ChromaDB의 거리(Distance) 값을 유사도 점수(Similarity Score)로 변환하여 반환 (Similarity = 1.0 - Distance).
  - 메타데이터 필터링 지원.
- **삭제**:
  - `delete`: ID 목록으로 삭제.
  - `delete_by_source`: 특정 파일(Source)에서 나온 모든 청크를 찾아 삭제하는 편의 기능 제공. 이는 파일 업데이트 시 이전 데이터를 정리하는 데 유용합니다.
- **메타데이터 관리**: 소스 파일 목록 추적 (`get_all_sources`) 등을 구현하여 인덱싱 상태 관리를 돕습니다.

### 3. FAISSVectorStore (Placeholder)

대규모 데이터 처리를 위한 FAISS 구현체 스켈레톤입니다. 현재는 `NotImplementedError`를 발생시키며, 추후 고성능 검색이 필요할 때 구현될 예정입니다.

## 싱글톤 패턴

`get_default_vector_store()` 함수를 통해 애플리케이션 전역에서 하나의 VectorStore 인스턴스를 공유하여 사용합니다. 이는 DB 연결 리소스를 효율적으로 관리하기 위함입니다.

## ChromaDB 선택 이유

- **개발 편의성**: 별도의 서버 구축 없이 로컬 파일로 동작.
- **메타데이터 필터링**: 강력한 메타데이터 쿼리 기능 내장.
- **문서 관리**: 문서 ID, 메타데이터, 벡터를 한 번에 관리하기 용이함.

## 데이터 흐름

1. **저장(Add)**: `DocumentChunk` 리스트 -> ID/Vector/Text/Metadata 분리 -> ChromaDB Upsert -> 디스크 저장 (`vector_store/chroma.sqlite3`)
2. **검색(Search)**: 쿼리 벡터 -> HNSW 인덱스 근사 탐색 -> 필터링 -> Top-k 결과 반환
