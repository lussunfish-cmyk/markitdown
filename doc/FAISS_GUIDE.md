# FAISS 벡터 저장소 가이드

## 개요

FAISS(Facebook AI Similarity Search)는 대규모 벡터 데이터에 최적화된 고성능 벡터 저장소입니다.

### 성능 비교 (10,000개 문서 @ 1024차원)

| 항목 | FAISS | ChromaDB | 배율 |
|-----|-------|---------|------|
| 추가 시간 | 3.8초 | 4.3초 | 1.1배 |
| 평균 검색 시간 | 1.455ms | 2.804ms | **1.9배 빠름** |
| QPS (초당 쿼리) | 687 q/s | 357 q/s | **1.9배** |
| 저장소 크기 | 40.69 MB | ≈80+ MB | 50-60% 더 효율 |

**결론**: 수십만 개 이상의 문서나 높은 QPS가 필요한 경우 FAISS 사용 권장.

---

## 1. FAISS 활성화

### 환경 변수 설정

```bash
# .env 파일 또는 셸 환경변수
export VECTOR_STORE_TYPE=faiss           # chroma에서 faiss로 변경
export FAISS_INDEX_TYPE=flat             # 인덱스 타입 선택
```

### 또는 코드에서 직접

```python
from app.vector_store import FAISSVectorStore
from pathlib import Path

store = FAISSVectorStore(
    collection_name="my_documents",
    persist_directory=Path("./vector_store"),
    index_type="flat"
)
```

---

## 2. 인덱스 타입 비교

### Flat (기본값) - 균형잡힌 선택
```
index_type = "flat"
```
- **메모리**: 40MB/10k docs @1024dim
- **검색 속도**: 1.455ms (10k docs)
- **추가 속도**: 3.8초 (10k docs)
- **추천**: 일반적인 사용 사례, ~1M 문서까지

### IVF (Inverted File Index) - 메모리 효율
```
index_type = "ivf"
```
- **메모리**: 35-40% 감소 (큰 데이터셋에서 효과적)
- **검색 속도**: 비슷 (nprobe 설정에 따라 가변)
- **추천**: 매우 큰 데이터셋 (10M+)
- **주의**: 최소 390개 이상의 훈련 데이터 필요 (자동 조정됨)

### HNSW (Hierarchical Navigable Small World) - 최고 속도
```
index_type = "hnsw"
```
- **메모리**: 가장 많음 (그래프 저장)
- **검색 속도**: Flat보다 약간 빠름
- **추가 속도**: 느림
- **추천**: 매우 큰 데이터셋 + 낮은 지연 요구

---

## 3. 환경변수 전체 설정

### 필수 설정 파일 (`.env`)

```bash
# 벡터 저장소 설정
VECTOR_STORE_TYPE=faiss                 # chroma 또는 faiss
VECTOR_STORE_DIR=./vector_store         # 저장 경로

# FAISS 특정 설정
FAISS_INDEX_TYPE=flat                   # flat, ivf, hnsw
FAISS_USE_GPU=false                     # true/false (현재 macOS 미지원)

# 임베딩 설정
EMBEDDING_DIM=1024                      # multilingual-e5-large 차원
```

---

## 4. 사용 예제

### 기본 사용

```python
from app.vector_store import get_vector_store
import numpy as np

# 저장소 초기화 (환경변수에서 자동으로 설정 로드)
store = get_vector_store()

# 문서 추가
ids = ["doc_1", "doc_2", "doc_3"]
embeddings = np.random.rand(3, 1024).tolist()
documents = ["내용 1", "내용 2", "내용 3"]
metadatas = [
    {"source": "file1.txt", "chunk_idx": 0},
    {"source": "file2.txt", "chunk_idx": 0},
    {"source": "file3.txt", "chunk_idx": 0}
]

store.add(ids, embeddings, documents, metadatas)

# 검색
query = np.random.rand(1024).tolist()
results = store.search(query, k=5)

for result in results:
    print(f"ID: {result['id']}, Score: {result['score']:.3f}")
    print(f"Content: {result['document'][:50]}...")
```

### 대규모 배치 추가

```python
# 효율적인 배치 처리 (메모리 절약)
BATCH_SIZE = 1000

for batch_start in range(0, total_docs, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, total_docs)
    batch_size = batch_end - batch_start
    
    # 배치 데이터 준비
    batch_ids = prepare_ids(batch_start, batch_end)
    batch_embeddings = generate_embeddings(batch_start, batch_end)
    batch_documents = get_documents(batch_start, batch_end)
    batch_metadatas = get_metadatas(batch_start, batch_end)
    
    # 저장소에 추가
    store.add(batch_ids, batch_embeddings, batch_documents, batch_metadatas)
    
    print(f"Processed {batch_end}/{total_docs}")
```

### 필터링 검색

```python
# 특정 소스 파일만 검색
results = store.search(query, k=10, filter={"source": "file1.txt"})

# 포스트 필터링 (현재 단일 필터만 지원)
for result in results:
    metadata = result['metadata']
    if metadata.get('chunk_idx') > 0:
        print(f"Filtered: {result['id']}")
```

### 컬렉션 정보

```python
info = store.get_collection_info()
print(f"총 문서 개수: {info['total_chunks']}")
print(f"임베딩 차원: {info['embedding_dim']}")
print(f"저장소 크기: {info['total_size']}")
print(f"인덱스 타입: {info['index_type']}")

# 고유 소스 목록
sources = store.get_all_sources()
print(f"소스 파일: {sources}")
```

### 삭제 및 초기화

```python
# 개별 문서 삭제
store.delete(["doc_1", "doc_2"])

# 소스 파일별 삭제
deleted = store.delete_by_source("file1.txt")
print(f"{deleted}개 문서 삭제됨")

# 전체 초기화
store.clear()
```

---

## 5. ChromaDB와 마이그레이션

### ChromaDB에서 FAISS로 변경

```bash
# 1. 환경변수 변경
export VECTOR_STORE_TYPE=faiss

# 2. 새로운 저장소가 자동으로 생성됨
# (기존 ChromaDB 데이터는 별도 디렉토리에 유지)
```

### 양쪽 동시 운영

```python
from app.vector_store import ChromaVectorStore, FAISSVectorStore
from pathlib import Path

# 두 저장소 모두 초기화
chroma_store = ChromaVectorStore(
    collection_name="docs_chroma",
    persist_directory=Path("./vector_store")
)

faiss_store = FAISSVectorStore(
    collection_name="docs_faiss",
    persist_directory=Path("./vector_store"),
    index_type="flat"
)

# ChromaDB에서 데이터 읽기
all_docs = chroma_store.get(ids)

# FAISS에 데이터 쓰기
faiss_store.add(ids, embeddings, documents, metadatas)
```

---

## 6. 성능 최적화 팁

### 대규모 데이터 추가 시

```python
# ✓ 권장: 배치 처리
for batch in get_batches(docs, batch_size=1000):
    store.add(batch_ids, batch_embeddings, batch_docs, batch_metas)

# ✗ 비권장: 개별 추가 (반복 오버헤드)
for doc in docs:
    store.add([doc_id], [embedding], [doc_text], [meta])
```

### 메모리 효율화

```python
# 임베딩을 리스트로 변환 전에 dtype 확인
embeddings = np.array(embeddings_list, dtype=np.float32)

# 필요없는 메타데이터 제거
# 큰 메타데이터는 SQLite 에 저장되므로 검색 시 I/O 증가
```

### 검색 최적화

```python
# IVF 인덱스 사용 시 nprobe 조정
# nprobe 클수록: 정확도 높음, 속도 낮음
# nprobe 작을수록: 빠름, 정확도 낮음
# 기본값: nlist의 1/10

# 예: nprobe를 명시적으로 설정 (내부 구현 필요)
store.index.nprobe = 20  # 기본: 10
```

---

## 7. 문제 해결

### 경고: "clustering N points to K centroids: please provide at least XXX training points"

**원인**: IVF 인덱스에서 훈련 데이터(K의 4배)가 부족

**해결책**:
```python
# 1. Flat 인덱스 사용 (권장)
store = FAISSVectorStore(index_type="flat")

# 2. IVF 사용 + 자동 조정 (코드에서 자동으로 처리됨)
store = FAISSVectorStore(index_type="ivf")  # nlist 자동 조정
```

### OOM (Out of Memory) 오류

**원인**: 너무 많은 문서를 한 번에 추가

**해결책**:
```python
# 배치 크기 감소
BATCH_SIZE = 100  # 기본 1000에서 감소

for batch_start in range(0, total, BATCH_SIZE):
    # 배치 처리
```

### 검색이 느림

**확인**:
```python
info = store.get_collection_info()
print(f"인덱스 타입: {info['index_type']}")
print(f"총 문서: {info['total_chunks']}")

# Flat 인덱스로 10M 이상 문서 저장 시 느릴 수 있음
# → IVF 또는 HNSW로 변경
```

---

## 8. 참고 자료

- [FAISS 공식 문서](https://github.com/facebookresearch/faiss/wiki)
- [벡터 데이터베이스 비교](https://github.com/erikbern/ann-benchmarks)
- [대규모 유사도 검색 튜토리얼](https://github.com/facebookresearch/faiss/tree/main/demos)
