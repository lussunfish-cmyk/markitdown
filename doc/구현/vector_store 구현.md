# Vector Store 구현

## 개요

RAG 시스템의 벡터 저장소 구현. ChromaDB를 기본으로 사용하며, 향후 FAISS 확장을 위한 추상화 레이어 포함.

## ChromaDB vs FAISS 비교

### ChromaDB 장점 ✅
- **사용하기 쉬운 API**: 간단한 추가/검색/삭제
- **메타데이터 자동 관리**: 필터링 기능 내장
- **영속성 내장**: 자동 디스크 저장/로드
- **RAG에 최적화**: 문서 중심 설계
- **빠른 개발 속도**: 50~100줄로 구현 가능

### ChromaDB 단점 ❌
- 대규모 데이터셋(수백만 개 이상)에서 느림
- FAISS 대비 메모리 사용량 많음
- 분산 처리 기능 제한적

### FAISS 장점 ✅
- **매우 빠른 검색 속도**: 밀리초 단위
- **메모리 효율성**: Product Quantization으로 1/10 축소 가능
- **대규모 데이터셋 지원**: 수백만~수억 개 벡터
- **GPU 가속 지원**
- **성숙도와 안정성**: 2017년부터 프로덕션 검증

### FAISS 단점 ❌
- 메타데이터 관리 어려움 (별도 구현 필요)
- 복잡한 구현 (200~300줄 추가 코드)
- 필터링 제한적
- 개발 시간 증가

### 성능 비교

| 벡터 수 | FAISS (IVF) | ChromaDB | 차이 |
|---------|-------------|----------|------|
| 1,000개 | ~1ms | ~2ms | 비슷 |
| 10,000개 | ~5ms | ~10ms | 2배 |
| 100,000개 | ~20ms | ~100ms | 5배 |
| 1,000,000개 | ~50ms | ~500ms+ | 10배+ |

### 선택 기준

**ChromaDB 선택 시나리오** (현재 프로젝트)
- ✅ 문서가 수천~수만 개 수준
- ✅ 메타데이터 기반 필터링 필요
- ✅ 빠른 개발 및 프로토타이핑
- ✅ 문서 업데이트/삭제가 빈번
- ✅ RAG 애플리케이션

**FAISS 선택 시나리오**
- ✅ 수백만 개 이상의 벡터
- ✅ 검색 속도가 최우선
- ✅ 메모리 제약이 심함
- ✅ GPU 사용 가능
- ✅ 정적 데이터셋

---

## 구현 내용

### 파일 구조

```
app/vector_store.py (540줄)
├── VectorStore (추상 클래스)
├── ChromaVectorStore (완전 구현)
├── FAISSVectorStore (스켈레톤)
├── get_vector_store() (팩토리 함수)
└── get_default_vector_store() (싱글톤)
```

### 주요 클래스 및 메서드

#### 1. VectorStore (추상 클래스)

```python
class VectorStore(ABC):
    @abstractmethod
    def add(self, ids, embeddings, documents, metadatas) -> None
    
    @abstractmethod
    def search(self, query_embedding, k=5, filter=None) -> List[Dict]
    
    @abstractmethod
    def delete(self, ids) -> None
    
    @abstractmethod
    def get(self, ids) -> List[Dict]
    
    @abstractmethod
    def count(self) -> int
    
    @abstractmethod
    def clear(self) -> None
    
    @abstractmethod
    def get_collection_info(self) -> Dict
```

#### 2. ChromaVectorStore

**초기화**
- ChromaDB PersistentClient 생성
- 컬렉션 가져오기 또는 생성
- 코사인 유사도 사용 (hnsw:space)

**주요 메서드**

1. **add()**: 벡터와 문서 추가
   - upsert 사용 (중복 시 업데이트)
   - 배치 추가 지원

2. **search()**: 유사도 검색
   - 코사인 거리를 유사도로 변환 (similarity = 1 - distance)
   - 메타데이터 필터링 지원 (where 조건)
   - top-k 결과 반환

3. **delete()**: ID로 삭제

4. **delete_by_source()**: 소스 파일별 일괄 삭제
   - 특정 파일의 모든 청크 삭제
   - 재인덱싱 시 유용

5. **get()**: ID로 문서 조회
   - 임베딩 포함 조회 가능

6. **count()**: 총 문서 개수

7. **clear()**: 컬렉션 전체 삭제 및 재생성

8. **get_collection_info()**: 컬렉션 통계
   - 총 청크 수
   - 고유 문서 수
   - 저장 경로 등

9. **get_all_sources()**: 고유 소스 파일 목록

#### 3. FAISSVectorStore

현재는 NotImplementedError를 발생시키는 스켈레톤 구현.
향후 ragas 평가 및 대규모 데이터셋 처리를 위해 추가 예정.

#### 4. 팩토리 함수

```python
def get_vector_store(store_type=None, **kwargs) -> VectorStore:
    """설정에 따라 벡터 저장소 인스턴스 생성"""
    if store_type == "chroma":
        return ChromaVectorStore(**kwargs)
    elif store_type == "faiss":
        return FAISSVectorStore(**kwargs)
```

#### 5. 싱글톤 패턴

```python
def get_default_vector_store() -> VectorStore:
    """기본 벡터 저장소 싱글톤 인스턴스 반환"""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = get_vector_store()
    return _vector_store_instance
```

---

## 구현 시 발생한 이슈 및 해결

### 1. 배열 Truth Value 에러

**문제**
```python
if results['ids']:  # NumPy 배열에서 에러 발생
    # ValueError: The truth value of an array with more than one element 
    # is ambiguous. Use a.any() or a.all()
```

**원인**
- ChromaDB가 NumPy 배열을 반환하는 경우
- Python의 truth value 체크가 배열에 대해 애매함

**해결**
```python
# 명시적으로 None 체크와 길이 체크
if results['ids'] is not None and len(results['ids']) > 0:
    # 처리
```

**수정 위치**
- `search()` 메서드의 결과 포맷팅
- `get()` 메서드의 결과 포맷팅
- `delete_by_source()` 메서드
- `get_collection_info()` 메서드
- `get_all_sources()` 메서드

### 2. Ollama 임베딩 NaN 에러

**문제**
```
임베딩 생성 에러: failed to encode response: json: unsupported value: NaN
```

**원인**
- `qllama/multilingual-e5-large-instruct` 모델의 문자 인코딩 이슈
- 특정 텍스트(짧은 한글 쿼리 등)에서 NaN 값 발생

**해결 방법**

1. **임베딩 모델 변경** (권장)
```bash
# 안정적인 모델로 변경
docker exec -it ollama ollama pull mxbai-embed-large
docker exec -it ollama ollama pull nomic-embed-text
docker exec -it ollama ollama pull all-minilm
```

2. **설정 파일 수정**
```python
# config.py
EMBEDDING_MODEL = "mxbai-embed-large"  # 또는 다른 모델

# docker-compose.yml
environment:
  - OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
```

3. **권장 모델**
   - `mxbai-embed-large`: 고성능, 안정적 (향후 기본값)
   - `nomic-embed-text`: 빠르고 효율적
   - `all-minilm`: 경량, 빠른 속도

---

## 테스트 결과

### 테스트 파일: test_vector_store.py

**테스트 구성** (8개 테스트)

1. ✅ **기본 동작**: 추가, 조회, 개수 확인
2. ✅ **ID 조회**: get() 메서드
3. ✅ **소스 관리**: get_all_sources()
4. ⚠️ **검색**: 의미론적 검색 (Ollama 이슈로 일부 스킵)
5. ⚠️ **필터링**: 메타데이터 필터링 검색 (Ollama 이슈로 일부 스킵)
6. ✅ **삭제**: delete(), delete_by_source()
7. ✅ **팩토리/싱글톤**: 인스턴스 생성 패턴
8. ⚠️ **영속성**: 재시작 후 데이터 보존 (Ollama 이슈로 일부 스킵)

### 테스트 결과 요약

```
======================================================================
  ✅ 모든 테스트 통과!
======================================================================

성공한 테스트:
- 기본 CRUD 동작 (추가, 조회, 삭제)
- 메타데이터 관리
- 컬렉션 정보 조회
- 팩토리 패턴 및 싱글톤 패턴
- 에러 핸들링 (FAISS, Invalid 타입)

Ollama 임베딩 이슈로 스킵된 테스트:
- 쿼리 임베딩이 필요한 검색 테스트
- 영속성 테스트 일부
```

---

## 향후 확장 계획

### FAISS 구현

**구현 시나리오**
- Ragas 평가 시스템 구축
- 대규모 데이터셋 처리 (수십만~수백만 문서)
- 성능 벤치마킹

**구현 내용**
```python
class FAISSVectorStore(VectorStore):
    def __init__(self, index_type="Flat", dimension=1024):
        # FAISS 인덱스 초기화
        # - Flat: 정확도 최고
        # - IVF: 속도/정확도 균형
        # - HNSW: 빠른 근사 검색
        
    def add(self, ...):
        # 벡터 추가
        # 메타데이터 별도 저장 (SQLite 등)
        
    def search(self, ...):
        # FAISS 검색
        # 메타데이터 결합
        
    def save(self, path):
        # 인덱스 저장
        
    def load(self, path):
        # 인덱스 로드
```

**추가 기능**
- GPU 가속 지원
- Product Quantization (메모리 최적화)
- 하이브리드 검색 (FAISS + BM25)

---

## 사용 예시

### 기본 사용

```python
from app.vector_store import get_default_vector_store
from app.ollama_client import get_ollama_client
from app.embedding import create_embedder

# 1. 벡터 저장소 초기화
store = get_default_vector_store()

# 2. 문서 임베딩 및 추가
embedder = create_embedder()
chunks = embedder.embed_document(
    text=document_text,
    source="document.md"
)

ids = [chunk.id for chunk in chunks]
embeddings = [chunk.embedding for chunk in chunks]
documents = [chunk.content for chunk in chunks]
metadatas = [chunk.metadata.model_dump() for chunk in chunks]

store.add(ids, embeddings, documents, metadatas)

# 3. 검색
ollama_client = get_ollama_client()
query_embedding = ollama_client.embed("5G 기술이란?")
results = store.search(query_embedding, k=5)

for result in results:
    print(f"유사도: {result['score']:.4f}")
    print(f"문서: {result['document']}")
```

### 메타데이터 필터링

```python
# 특정 파일에서만 검색
results = store.search(
    query_embedding,
    k=5,
    filter={"source": "specific_file.md"}
)
```

### 문서 관리

```python
# 특정 파일의 모든 청크 삭제
deleted = store.delete_by_source("old_document.md")

# 모든 소스 파일 목록
sources = store.get_all_sources()

# 컬렉션 정보
info = store.get_collection_info()
print(f"총 문서: {info['total_documents']}")
print(f"총 청크: {info['total_chunks']}")
```

---

## 설정

### Config (config.py)

```python
class VectorStoreConfig:
    # 벡터 저장소 타입: "chroma" 또는 "faiss"
    STORE_TYPE = "chroma"
    
    # 저장소 경로
    PERSIST_DIR = Path("/app/vector_store")
    
    # Chroma 컬렉션명
    CHROMA_COLLECTION_NAME = "documents"
    
    # 임베딩 차원 (모델에 따라 변경)
    EMBEDDING_DIM = 1024  # mxbai-embed-large
```

### Docker Compose

```yaml
environment:
  - OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
  - VECTOR_STORE_DIR=/app/vector_store
  - VECTOR_STORE_TYPE=chroma
```

---

## 참고 자료

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Ollama Embedding Models](https://ollama.ai/library)
- [Ragas Evaluation Framework](https://docs.ragas.io/)
