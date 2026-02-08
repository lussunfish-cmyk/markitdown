# Vector Store 구현

## 개요

임베딩된 문서 벡터를 저장하고 검색하는 저장소(Vector Database) 인터페이스 및 구현체입니다. 현재는 ChromaDB를 메인으로 지원합니다.

## 파일 경로

```
markitdown/app/vector_store.py
```

## 주요 클래스

### 1. VectorStore (Abstract)
벡터 저장소의 추상 기본 클래스입니다.
- `add`, `search`, `delete`, `get`, `count`, `clear` 등의 메서드를 정의합니다.

### 2. ChromaVectorStore
ChromaDB를 사용하는 구현체입니다.
- **PersistentClient**: 데이터를 디스크(`vector_store/`)에 영구 저장합니다.
- **Collection**: 문서를 저장하는 논리적 단위입니다.
- **기능**:
  - `add`: 문서, 임베딩, 메타데이터를 저장합니다. (Upsert 방식)
  - `search`: 코사인 유사도 등을 사용하여 쿼리 벡터와 가장 유사한 문서를 검색합니다. 메타데이터 필터링을 지원합니다.
  - `delete_by_source`: 특정 파일(source)에 해당하는 모든 청크를 삭제합니다.
  - `get_collection_info`: 저장된 문서 수 등 통계 정보를 제공합니다.

### 3. FAISSVectorStore
FAISS 라이브러리를 사용하는 구현체 (현재 미구현, 인터페이스만 존재). 대규모 데이터셋이나 인메모리 검색이 필요할 때 확장 가능합니다.

## 팩토리 함수
- `get_vector_store`: 설정(`config.py`)에 따라 적절한 VectorStore 인스턴스(싱글톤)를 반환합니다.