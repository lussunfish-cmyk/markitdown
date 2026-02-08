# rag 구현

## 개요

RAG (Retrieval-Augmented Generation) 파이프라인의 핵심 모듈입니다. 사용자의 질문을 받아 검색(Retriever), 컨텍스트 구성, 프롬프트 생성, 그리고 LLM 답변 생성 과정을 통합적으로 제어합니다. 캐싱과 메트릭 측정을 통해 성능과 효율성을 관리합니다.

## 파일 경로

```
markitdown/app/rag.py
```

## 주요 클래스

### 1. RAGPipeline
RAG 전체 프로세스를 조율하는 컨트롤러 클래스입니다.

- **초기화**: `OllamaClient`, `VectorStore`, `Retriever` 등 필요한 컴포넌트를 설정에 맞춰 초기화합니다.
- **주요 메서드 `query(...)`**:
  1. **Query Rewriting**: 설정 시 LLM을 사용해 검색에 더 적합한 형태로 질의를 재작성합니다.
  2. **Retrieval**: `Retriever`를 사용해 관련 문서 검색. 캐시(`TTLCache`)가 적용되어 동일한 검색에 대해 빠르게 응답합니다.
  3. **Filtering**: 검색 결과 중 유사도가 너무 낮거나 품질이 떨어지는 청크를 필터링합니다.
  4. **Context Building**: 선택된 청크들을 포맷팅하여 LLM 프롬프트에 들어갈 컨텍스트 문자열을 생성합니다.
  5. **Prompting**: 시스템 프롬프트, 컨텍스트, 사용자 질문을 조합하여 최종 프롬프트를 완성합니다.
  6. **Generation**: `OllamaClient`를 통해 답변 생성.
  7. **Result Packaging**: 답변, 참조 소스, 사용된 컨텍스트, 메트릭 등을 `RAGResult` 객체로 반환합니다.

### 2. RAGMetrics
RAG 파이프라인의 각 단계별 소요 시간과 정보를 기록합니다.
- `query_time`, `embedding_time`, `search_time`, `llm_time`
- `num_chunks`, `context_length`, `cache_hit`

### 3. RAGResult
최종 반환되는 결과 구조체입니다.
- `answer`: 생성된 답변 텍스트.
- `sources`: 참조한 문서의 출처 정보 리스트.
- `context_used`: 실제 프롬프트에 사용된 컨텍스트 원문.
- `metrics`: 성능 측정 데이터.

## 주요 특징

- **캐싱**: 검색 결과에 대해 TTL(Time To Live) 캐시를 적용하여 반복적인 질문에 대한 응답 속도를 개선했습니다.
- **가시성**: 각 단계의 소요 시간과 캐시 적중 여부 등을 메트릭으로 제공하여 성능 튜닝이 용이합니다.
- **유연성**: 검색기 타입, Top-K, LLM 파라미터 등을 동적으로 조절할 수 있습니다.

## 데이터 흐름

1. 사용자 질문 -> `query()` 호출
2. (옵션) 쿼리 재작성 (LLM)
3. `Retriever`로 검색 (Vector + Keyword) -> 문서 청크 목록
4. 청크 필터링 및 Reranking
5. 시스템 프롬프트 + 청크 컨텍스트 + 질문 조합
6. `OllamaClient`로 LLM 생성 요청
7. 최종 답변 및 메타데이터 반환
