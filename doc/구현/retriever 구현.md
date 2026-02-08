# Retriever 구현

## 개요

문서 검색을 담당하는 모듈입니다. 단순한 벡터 검색을 넘어, 키워드 검색(BM25)과 결합한 하이브리드 검색, 그리고 검색 결과의 순위를 재조정하는 리랭킹(Reranking) 기능을 제공합니다.

## 파일 경로

```
markitdown/app/retriever.py
```

## 주요 클래스

### 1. BaseRetriever (Abstract)
모든 검색기의 기본 인터페이스를 정의합니다. `search(query, k)` 메서드를 반드시 구현해야 합니다.

### 2. VectorRetriever
임베딩 기반의 의미론적 검색(Semantic Search)을 수행합니다.
- `OllamaClient`를 통해 쿼리 임베딩을 생성하고, `VectorStore`에서 유사한 벡터를 찾습니다.

### 3. BM25Retriever
키워드 기반의 어휘적 검색(Lexical Search)을 수행합니다.
- `rank_bm25` 라이브러리를 사용합니다.
- 인덱싱된 문서들의 텍스트를 토큰화하여 BM25 인덱스를 메모리에 구축합니다.
- `_tokenize` 메서드에서 구두점을 제거하는 등 전처리를 수행합니다.

### 4. HybridRetriever
벡터 검색과 BM25 검색 결과를 결합합니다.
- **Weighted Fusion**: 두 검색 결과의 점수를 정규화한 후 가중치(`alpha`)를 적용하여 합산합니다.
- **RRF (Reciprocal Rank Fusion)**: 점수 대신 순위(Rank)를 기반으로 결과를 결합하여, 점수 스케일이 다른 두 검색 방식의 조화를 꾀합니다.

### 5. Reranker
1차 검색된 결과의 순위를 정밀하게 재조정합니다.
- **Cross-Encoder**: `sentence-transformers`의 Cross-Encoder 모델(예: `BAAI/bge-reranker-v2-m3`)을 사용하여 쿼리와 문서 간의 관련성을 직접 채점합니다. 가장 정확도가 높습니다.
- **OOM 방지**: GPU 메모리 부족을 방지하기 위해 배치 사이즈를 조절하고 캐시를 비우는 로직이 포함되어 있습니다.
- **Fallback**: Cross-Encoder 로딩 실패 시 BM25 등을 대체재로 사용할 수 있습니다.

### 6. AdvancedRetriever
하이브리드 검색과 리랭킹을 모두 수행하는 최상위 검색기입니다.
- 1단계: `HybridRetriever`로 후보군(Candidate)을 넉넉하게(`k * 10`) 추출합니다.
- 2단계: `Reranker`로 후보군을 재정렬하여 최종 상위 `k`개를 반환합니다.

## 데이터 흐름 (AdvancedRetriever 기준)

1. `search(query, k)` 호출.
2. `HybridRetriever`가 벡터 검색과 BM25 검색을 병렬 수행.
3. 검색 결과를 RRF 또는 가중치 합산으로 결합하여 상위 `N`개 후보 추출.
4. `Reranker`가 후보 문서들에 대해 Cross-Encoder 점수 계산.
5. 점수순으로 재정렬하여 상위 `k`개 반환.