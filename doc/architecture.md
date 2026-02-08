# 마크다운 변환 + RAG 시스템 아키텍쳐

## 구현 상태 (2026-02-07 기준)

✅ **Phase 1 완료**: 모든 핵심 모듈 구현 완료  
✅ **Phase 2 완료**: 배치 처리 및 통합 API 구현 완료  
✅ **Phase 3 완료**: 에러 수정 및 최적화 완료

---

## 핵심 모듈 (구현 완료 ✅)

### 1. **`config.py`** ✅
**상태**: 구현 완료 (BatchConfig 포함)

**주요 기능**:
- Ollama 설정 (URL, 모델명)
- 벡터 DB 설정 (ChromaDB)
- 청킹 파라미터 (chunk_size=512, overlap=128)
- RAG 하이퍼파라미터 (top_k, temperature)
- 배치 처리 설정 (BATCH_STATE_DIR, DEFAULT_BATCH_SIZE=100)

**구현 내용**:
```python
class OllamaConfig:
    BASE_URL = "http://ollama:11434"
    LLM_MODEL = "gemma2"
    EMBEDDING_MODEL = "mxbai-embed-large"

class ChunkingConfig:
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 128

class VectorStoreConfig:
    PERSIST_DIRECTORY = "/app/vector_store"
    COLLECTION_NAME = "markdown_documents"

class BatchConfig:
    BATCH_STATE_DIR = "/app/batch_state"
    DEFAULT_BATCH_SIZE = 100
```

**참고**: [config 구현.md](./구현/config%20구현.md)

---

### 2. **`schemas.py`** ✅
**상태**: 구현 완료 (Pydantic V2 호환)

**주요 기능**:
- Pydantic 모델 정의
- RAG 요청/응답 스키마
- 배치 처리 스키마
- 인덱싱 관련 데이터 모델

**구현된 스키마**:
- `ConvertResponse`, `FileInfo`
- `BatchJobResponse`, `BatchInfo`, `BatchFileStatus`
- `IndexFileRequest`, `IndexFolderRequest`
- `RAGRequest`, `RAGResponse`, `SearchRequest`, `RetrievalResult`

**참고**: [schemas 구현.md](./구현/schemas%20구현.md)

---

### 3. **`ollama_client.py`** ✅
**상태**: 구현 완료

**주요 기능**:
- Ollama API와 통신하는 클라이언트
- gemma2 모델 호출 (completion)
- mxbai-embed-large 모델로 embedding 생성
- 연결 관리 및 에러 핸들링
- 스트리밍 응답 지원

**참고**: [ollama_client 구현.md](./구현/ollama_client%20구현.md)

---

### 4. **`embedding.py`** ✅
**상태**: 구현 완료

**주요 기능**:
- 마크다운 문서를 벡터로 변환
- 텍스트 청킹 (512자 단위, 128자 오버랩)
- 메타데이터 추출 및 관리
- 배치 임베딩 처리

**참고**: [embedding 구현.md](./구현/embedding%20구현.md)

---

### 5. **`vector_store.py`** ✅
**상태**: 구현 완료 (ChromaDB 사용)

**주요 기능**:
- ChromaDB 벡터 DB 관리
- 임베딩 저장/로드
- 유사도 검색 (similarity search)
- 인덱스 관리 (추가/삭제/업데이트)
- 현재 1322개 문서 인덱싱됨

**참고**: [vector_store 구현.md](./구현/vector_store%20구현.md)

---

### 6. **`retriever.py`** ✅
**상태**: 구현 완료 (Hybrid Search + Reranking)

**주요 기능**:
- **AdvancedRetriever**: 벡터 + 키워드 검색 결합
- **Hybrid Search**: Semantic(Vector) + Lexical(BM25)
- **Reciprocal Rank Fusion (RRF)**: 검색 결과의 순위 기반 통합
- **Reranker**: BAAI/bge-reranker-v2-m3 사용하여 최종 순위 재조정
- search() 메서드로 고품질 문서 추출

**참고**: [retriever 구현.md](./구현/retriever%20구현.md)

---

### 7. **`rag.py`** ✅
**상태**: 구현 완료 (Phase 2 개선 적용)

**주요 기능**:
- RAGPipeline 클래스
- **Query Rewriting**: LLM을 이용한 검색 쿼리 최적화
- **Cache System**: 검색 결과 및 임베딩 캐싱 (TTL 적용)
- **Performance Metrics**: 검색/생성 시간, 청크 사용량 등 측정
- gemma2를 통한 답변 생성 및 프롬프트 관리

**참고**: [rag 구현.md](./구현/rag%20구현.md)

---

### 8. **`indexer.py`** ✅
**상태**: 구현 완료

**주요 기능**:
- DocumentIndexer 클래스
- output 폴더의 마크다운 파일 자동 인덱싱
- 배치 인덱싱 기능
- 인덱싱 상태 관리
- 0.1초 딜레이로 ChromaDB 동시성 문제 해결

**참고**: [indexer 구현.md](./구현/indexer%20구현.md)

---

### 9. **`batch_manager.py`** ✅ (신규 추가)
**상태**: 구현 완료

**주요 기능**:
- BatchStateManager 클래스
- 대량 파일 배치 처리 상태 관리
- JSON 기반 체크포인트 시스템
- 중단/재시작 지원
- 진행률 추적

**핵심 메서드**:
- `create_batch()` - 배치 작업 생성
- `update_file_status()` - 파일별 상태 업데이트
- `update_batch_status()` - 배치별 상태 업데이트
- `load_state()` / `save_state()` - 영구 저장
- `get_next_pending_batch()` - 재시작 지원

**참고**: [batch_manager 구현.md](./구현/batch_manager%20구현.md)

---

### 10. **`converter.py`** ✅
**상태**: 통합 완료 (15개 엔드포인트)

**주요 기능**:
- FastAPI 통합 서비스
- 변환 + 임베딩 + RAG를 하나의 앱으로 통합

**구현된 엔드포인트**:

#### 변환 API
- `POST /convert` - 단일 파일 변환 (auto_index 옵션)
- `POST /convert-folder` - 폴더 전체 변환 (auto_index 옵션)
- `POST /convert-batch` - 파일 업로드 배치 변환

#### 배치 처리 API
- `POST /batch/folder` - 서버 폴더 배치 처리
- `GET /batch/{batch_id}` - 배치 상태 조회
- `DELETE /batch/{batch_id}` - 배치 삭제
- `GET /batch` - 전체 배치 목록

#### 인덱싱 API
- `POST /index` - 단일 파일 인덱싱
- `POST /index-folder` - 폴더 전체 인덱싱

#### RAG API
- `POST /query` - RAG 질의응답
- `GET /search` - 유사 문서 검색
- `GET /documents` - 인덱싱된 문서 목록

**참고**: [convert 구현.md](./구현/convert%20구현.md)

---

## 아키텍쳐 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI (converter.py)                  │
│  - 15개 REST API 엔드포인트                                   │
│  - 변환 + 배치 처리 + 인덱싱 + RAG 통합                       │
└───────────┬─────────────────────────────────────────────────┘
            │
            ├─────► batch_manager.py (배치 상태 관리)
            │        └─► JSON 파일 영구 저장
            │
            ├─────► indexer.py (문서 인덱싱)
            │        ├─► embedding.py (청킹 + 임베딩)
            │        │    └─► ollama_client.py (mxbai-embed-large)
            │        └─► vector_store.py (ChromaDB 저장)
            │
            └─────► rag.py (RAG 파이프라인)
                     ├─► retriever.py (문서 검색)
                     │    └─► vector_store.py (유사도 검색)
                     └─► ollama_client.py (gemma2 답변 생성)
```

---

## 데이터 흐름

### 1. 변환 + 자동 인덱싱
```
PDF/DOCX/PPTX 파일
    ↓
markitdown 변환
    ↓
Markdown 파일 (output/)
    ↓ (auto_index=True)
청킹 (512자, 128 오버랩)
    ↓
Ollama 임베딩 (mxbai-embed-large)
    ↓
ChromaDB 저장
```

### 2. 배치 처리
```
300개 파일
    ↓
BatchStateManager.create_batch()
    ↓
100개씩 3개 배치로 분할
    ↓
각 배치 순차 처리
    ├─► 파일별 상태 저장 (JSON)
    └─► 진행률 추적
    ↓
전체 완료
```

### 3. RAG 쿼리
```
사용자 질문
    ↓
쿼리 재작성 (Optional)
    ↓
임베딩 생성 (쿼리) + 키워드 추출
    ↓
Hybrid Search
  ├─► ChromaDB (Vector Search): 의미적 유사성
  └─► BM25 (Keyword Search): 정확한 키워드 매칭
    ↓
Rank Fusion (Weighted / RRF)
    ↓
Reranking (BGE-M3) - 상위 문서 재정렬
    ↓
컨텍스트 구성
    ↓
gemma2 모델 호출
    ↓
답변 생성 (출처 포함)
```






---

## 기술 스택

### 핵심 기술
- **Python 3.10+**: 메인 언어
- **FastAPI**: REST API 프레임워크
- **ChromaDB**: 벡터 저장소
- **Ollama**: LLM 및 임베딩 (gemma2, mxbai-embed-large)
- **Rank_BM25**: 키워드 검색 알고리즘
- **Sentence-Transformers**: Cross-Encoder 리랭킹 (BGE-M3)

### 인프라
- **Docker Compose**: 컨테이너 오케스트레이션
- **Volume 마운트**: 데이터 영구 보존
  - `/app/input` - 입력 파일
  - `/app/output` - 변환 결과
  - `/app/vector_store` - ChromaDB 저장소
  - `/app/batch_state` - 배치 작업 상태

---

## 구현 완료 이력

### ✅ Phase 1: 핵심 모듈 (완료)
```
1️⃣ config.py           ✅ BatchConfig 추가
2️⃣ schemas.py          ✅ 배치/인덱싱/RAG 스키마 추가
3️⃣ ollama_client.py    ✅ Ollama 통신 구현
4️⃣ embedding.py        ✅ 청킹 + 임베딩 구현
5️⃣ vector_store.py     ✅ ChromaDB 연동
6️⃣ retriever.py        ✅ 검색 기능 구현
7️⃣ rag.py              ✅ RAG 파이프라인 구현
8️⃣ indexer.py          ✅ 자동 인덱싱 구현
9️⃣ batch_manager.py    ✅ 배치 상태 관리 (신규)
🔟 converter.py         ✅ 15개 엔드포인트 통합
```

### ✅ Phase 2: 최적화 및 에러 수정 (완료)

#### Pydantic V2 호환성
- ❌ `.dict()` → ✅ `.model_dump()` 전역 수정
- 영향 파일: converter.py, indexer.py, batch_manager.py

#### API 파라미터 통일
- ❌ `RAGPipeline.query(query=...)` → ✅ `RAGPipeline.query(question=...)`
- ❌ `retriever.retrieve()` → ✅ `retriever.search()`

#### ChromaDB 동시성 문제
- ❌ `readonly database` 에러 발생
- ✅ 인덱싱 시 0.1초 딜레이 추가
- ✅ 상세 로깅 추가

#### 임베딩 길이 최적화
- ❌ `context length exceeded` 에러
- ✅ CHUNK_SIZE: 1024 → 512 (축소)
- ✅ CHUNK_OVERLAP: 256 → 128 (축소)

---

## 주요 설정값

### 청킹 설정
```python
CHUNK_SIZE = 512      # 한 청크당 문자 수
CHUNK_OVERLAP = 128   # 청크 간 오버랩
```

### 배치 설정
```python
DEFAULT_BATCH_SIZE = 100    # 배치당 파일 수
BATCH_STATE_DIR = "/app/batch_state"
```

### RAG 설정
```python
TOP_K = 5                  # 검색 결과 개수
TEMPERATURE = 0.7          # LLM 온도
MAX_TOKENS = 2000          # 최대 토큰 수
```

### Ollama 모델
```python
LLM_MODEL = "gemma2"              # 답변 생성 모델
EMBEDDING_MODEL = "mxbai-embed-large"  # 임베딩 모델
```

---

## 디렉토리 구조

```
markitdown/
├── app/
│   ├── config.py              ✅ 전체 설정
│   ├── schemas.py             ✅ Pydantic 모델
│   ├── ollama_client.py       ✅ Ollama 클라이언트
│   ├── embedding.py           ✅ 임베딩 생성
│   ├── vector_store.py        ✅ ChromaDB 관리
│   ├── retriever.py           ✅ 문서 검색
│   ├── rag.py                 ✅ RAG 파이프라인
│   ├── indexer.py             ✅ 문서 인덱싱
│   ├── batch_manager.py       ✅ 배치 상태 관리
│   └── converter.py           ✅ FastAPI 통합
├── doc/
│   ├── 아키텍쳐.md             ✅ 전체 설계 문서
│   ├── USAGE.md               📖 사용 가이드
│   └── 구현/
│       ├── config 구현.md
│       ├── schemas 구현.md
│       ├── ollama_client 구현.md
│       ├── embedding 구현.md
│       ├── vector_store 구현.md
│       ├── retriever 구현.md
│       ├── rag 구현.md
│       ├── indexer 구현.md
│       ├── batch_manager 구현.md  ✅ 신규
│       └── convert 구현.md
├── input/                     📁 입력 파일
├── output/                    📁 변환된 마크다운
├── vector_store/              📁 ChromaDB 데이터
├── batch_state/               📁 배치 작업 상태 (JSON)
├── docker-compose.yml         🐳 컨테이너 설정
├── Dockerfile                 🐳 이미지 정의
└── requirements.txt           📦 의존성
```

---

## API 엔드포인트 목록

### 변환 API
| Method | Endpoint | 설명 | auto_index |
|--------|----------|------|------------|
| POST | `/convert` | 단일 파일 변환 | ✅ |
| POST | `/convert-folder` | 폴더 전체 변환 | ✅ |
| POST | `/convert-batch` | 파일 업로드 배치 | ✅ |

### 배치 처리 API
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/batch/folder` | 서버 폴더 배치 처리 |
| GET | `/batch/{batch_id}` | 배치 상태 조회 |
| DELETE | `/batch/{batch_id}` | 배치 삭제 |
| GET | `/batch` | 전체 배치 목록 |

### 인덱싱 API
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/index` | 단일 파일 인덱싱 |
| POST | `/index-folder` | 폴더 전체 인덱싱 |

### RAG API
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/query` | RAG 질의응답 |
| GET | `/search` | 유사 문서 검색 |
| GET | `/documents` | 인덱싱된 문서 목록 |

---

## 테스트 현황

### 단위 테스트
- ✅ `test_basic.py` - 기본 변환 기능
- ✅ `test_embedding.py` - 임베딩 생성
- ✅ `test_vector_store.py` - ChromaDB 저장/검색
- ✅ `test_retriever.py` - 문서 검색
- ✅ `test_indexer.py` - 인덱싱
- ✅ `test_rag.py` - RAG 파이프라인

### 통합 테스트
- ✅ 변환 + 자동 인덱싱 (auto_index=True)
- ✅ 배치 처리 (100개 단위)
- ✅ RAG 쿼리 (검색 + 답변 생성)
- ✅ 1322개 문서 인덱싱 검증

---

## 성능 지표

### 처리 속도
- 단일 파일 변환: ~2-5초
- 100개 배치 처리: ~10-15분
- 인덱싱 (파일당): ~0.5-1초
- RAG 쿼리 응답: ~2-3초

### 리소스 사용
- ChromaDB 저장소: 1322개 문서 = ~50MB
- 배치 상태 파일: 파일당 ~0.5KB
- 메모리: ~2-4GB (Ollama 포함)

---

## 향후 개선 사항

### 성능 최적화
- [ ] 배치 내 병렬 처리 (asyncio)
- [ ] 임베딩 캐싱
- [ ] 청크 크기 동적 조정

### 기능 추가
- [ ] 하이브리드 검색 (벡터 + 키워드)
- [ ] 리랭킹 (re-ranking)
- [ ] 문서 버전 관리

### 모니터링
- [ ] 프로메테우스 메트릭
- [ ] 로그 집계 (ELK)
- [ ] 헬스 체크 엔드포인트

---

## 문제 해결

### 일반적인 에러

#### 1. Ollama 연결 실패
```bash
# Ollama 서버 확인
docker compose ps
curl http://localhost:11434/api/tags
```

#### 2. ChromaDB readonly 에러
- **원인**: 동시성 문제
- **해결**: 인덱싱 시 0.1초 딜레이 추가됨

#### 3. Embedding 길이 초과
- **원인**: CHUNK_SIZE가 너무 큼
- **해결**: CHUNK_SIZE를 512로 축소

#### 4. Pydantic V2 경고
- **원인**: `.dict()` 메서드 deprecated
- **해결**: `.model_dump()` 사용

---

## 참고 문서

- [USAGE.md](./USAGE.md) - 사용 방법
- [구현 문서](./구현/) - 모듈별 상세 문서
- [테스트 방법](./테스트%20방법.md) - 테스트 가이드