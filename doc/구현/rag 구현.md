# RAG 구현

## 개요

RAG(Retrieval-Augmented Generation) 파이프라인을 구현한 모듈입니다. 사용자 질문에 대해 관련 문서를 검색하고, 이를 컨텍스트로 구성하여 LLM(Ollama)을 통해 답변을 생성합니다. 검색 품질 향상을 위한 다양한 기법(Query Rewriting, Caching, Reranking 등)이 적용되어 있습니다.

## 파일 경로

```
markitdown/app/rag.py
```

## 주요 클래스

### 1. RAGPipeline

RAG 프로세스의 전체 흐름을 제어하는 핵심 클래스입니다.

- **주요 기능**:
  - **Query Rewriting**: `_rewrite_query` 메서드를 통해 사용자 질문을 검색에 최적화된 형태로 변환합니다. (설정으로 활성화 가능)
  - **Caching**: `_search_with_cache` 메서드에서 `TTLCache`를 사용하여 동일한 검색 요청에 대한 결과를 캐싱합니다.
  - **Retrieval**: `AdvancedRetriever`를 사용하여 문서를 검색합니다. 쿼리 확장(Query Expansion)을 통해 원본 질문과 재작성된 질문을 모두 사용하여 검색 범위를 넓힙니다.
  - **Filtering**: `_filter_low_quality_results` 메서드로 유사도가 낮은 문서를 제외합니다.
  - **Context Building**: `_build_context` 메서드에서 검색된 문서를 LLM 프롬프트용 컨텍스트로 구성합니다. "Lost in the Middle" 현상을 방지하기 위해 중요 문서를 양 끝에 배치하는 재정렬 로직이 포함되어 있습니다.
  - **Generation**: `_generate_answer` 또는 `_stream_answer`를 통해 LLM 답변을 생성합니다.
  - **Metrics**: `RAGMetrics`를 통해 각 단계별 소요 시간과 사용된 청크 수 등을 측정합니다.

### 2. RAGMetrics & RAGResult

- **RAGMetrics**: 쿼리 처리 시간, 임베딩 시간, 검색 시간, LLM 생성 시간 등을 기록하는 데이터 클래스입니다.
- **RAGResult**: 최종 답변, 참조한 소스(Sources), 사용된 컨텍스트, 메타데이터 등을 포함하는 결과 객체입니다.

## 데이터 흐름

1. **질문 입력**: `query()` 메서드 호출.
2. **쿼리 확장**: 원본 질문 + 재작성된 쿼리(Query Rewriting) 준비.
3. **검색**: `Retriever`를 통해 관련 문서 청크 검색 (캐싱 적용).
4. **병합 및 정렬**: 다중 쿼리 검색 결과를 병합하고 점수순 정렬.
5. **필터링**: 품질이 낮은 결과 제외.
6. **컨텍스트 구성**: 검색된 청크들을 프롬프트에 맞게 포맷팅 및 재배치.
7. **프롬프트 생성**: 시스템 프롬프트 + 컨텍스트 + 질문 결합.
8. **답변 생성**: Ollama LLM 호출 (Completion).
9. **결과 반환**: 답변 및 출처 정보 반환.

## 주요 특징

- **성능 최적화**: 캐싱 및 비동기 처리를 고려한 설계.
- **정확도 향상**: Query Rewriting, Hybrid Search, Reranking, Context Reordering 등 최신 RAG 기법 적용.
- **관측성**: 상세한 로깅 및 메트릭 제공.