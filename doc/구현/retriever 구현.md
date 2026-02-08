# retriever 구현

## 개요

사용자의 자연어 질의(Query)에 대해 가장 관련성이 높은 문서 청크를 찾아오는 검색(Retrieval) 모듈입니다. 벡터 유사도 검색과 키워드 검색을 결합한 하이브리드 검색, 그리고 검색 결과를 재정렬하는 Reranking 기능을 제공하여 검색 품질을 극대화합니다.

## 파일 경로

```
markitdown/app/retriever.py
```

## 주요 클래스

### 1. BaseRetriever (Interface)
- 모든 검색기가 구현해야 할 `search(query, k)` 메서드를 정의합니다.

### 2. VectorRetriever
- **기능**: 임베딩 기반 의미론적(Semantic) 검색.
- **동작**:
  1. 질의를 임베딩 벡터로 변환 (`OllamaClient`).
  2. `VectorStore`에서 코사인 유사도가 높은 청크 검색.
- **특징**: 의미가 유사한 문서를 잘 찾지만, 정확한 키워드 매칭에는 약할 수 있습니다.

### 3. BM25Retriever
- **기능**: 키워드 빈도 기반(Tf-Idf 변형) 텍스트 검색.
- **동작**:
  1. 인덱싱된 문서들을 토큰화하여 BM25 인덱스 구축 (메모리 상).
  2. 질의 토큰과 매칭되는 문서 검색.
- **특징**: 고유명사나 정확한 용어 검색에 강합니다.

### 4. HybridRetriever
- **기능**: Vector 검색과 BM25 검색의 결과를 결합.
- **결합 방식 (Fusion Method)**:
  - **Weighted**: 점수 가중치 합 (Vector * alpha + BM25 * (1-alpha)).
  - **RRF (Reciprocal Rank Fusion)**: 순위 역수 합 (1 / (k + rank)). 설정에 따라 선택 가능.
- **전략**: 각 검색기에서 더 많은 후보(`k*3`개)를 가져온 뒤 결합하고 상위 `k`개를 반환합니다.

## 검색 결과 구조

- **SearchResult**: ID, Content, Score, Metadata를 포함하는 통일된 결과 객체입니다.

## 리랭킹 (Reranking)

- 검색된 결과의 순위를 더 정교한 모델(Cross-Encoder 등)을 사용하여 재조정하는 로직이 config에 따라 활성화될 수 있습니다 (구현상 `config.RETRIEVER.USE_RERANKER` 확인).

## 데이터 흐름

1. 질의(Query) 입력
2. 병렬 실행:
   - **VectorRetriever**: 임베딩 생성 -> ChromaDB 검색
   - **BM25Retriever**: 토큰화 -> BM25 점수 계산
3. 검색 결과 결합 (Weighted Fusion or RRF)
4. 상위 K개 후보 선정
5. (옵션) Reranker가 최종 순위 조정
6. 결과 반환 (SearchResult List)
