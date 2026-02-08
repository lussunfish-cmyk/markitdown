"""
RAG (검색 증강 생성) 파이프라인 구현.
질의응답을 위한 검색, 컨텍스트 구성, 답변 생성 통합 모듈.

Phase 2 성능 개선:
- 캐싱: 쿼리 임베딩 및 검색 결과 캐싱
- 스트리밍: 실시간 답변 생성
- 메트릭: 성능 측정 및 모니터링
- 프롬프트 최적화: 간결하고 효과적인 프롬프트
"""

import logging
import hashlib
import time
from typing import List, Optional, Dict, Any, Generator
from dataclasses import dataclass, field
from functools import lru_cache
from cachetools import TTLCache

from .config import config
from .ollama_client import get_ollama_client
from .vector_store import get_vector_store
from .retriever import get_retriever, SearchResult
from .schemas import RAGRequest, RAGResponse

logger = logging.getLogger(__name__)


# ============================================================================
# 캐시 설정
# ============================================================================

# 검색 결과 캐시 (설정에서 가져옴)
_search_cache = TTLCache(
    maxsize=config.CACHE.SEARCH_CACHE_MAXSIZE,
    ttl=config.CACHE.SEARCH_CACHE_TTL
)
_search_cache_hits = 0
_search_cache_misses = 0


# ============================================================================
# 프롬프트 템플릿 (Phase 2: 최적화됨)
# ============================================================================

# 설정 파일로 이동됨 (config.PROMPT)


# ============================================================================
# 성능 메트릭
# ============================================================================

@dataclass
class RAGMetrics:
    """RAG 파이프라인 성능 메트릭."""
    
    query_time: float = 0.0
    """전체 쿼리 처리 시간 (초)"""
    
    embedding_time: float = 0.0
    """임베딩 생성 시간 (초)"""
    
    search_time: float = 0.0
    """문서 검색 시간 (초)"""
    
    llm_time: float = 0.0
    """LLM 답변 생성 시간 (초)"""
    
    num_chunks: int = 0
    """사용된 청크 개수"""
    
    context_length: int = 0
    """컨텍스트 길이 (문자)"""
    
    cache_hit: bool = False
    """캐시 히트 여부"""
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "query_time": round(self.query_time, 3),
            "embedding_time": round(self.embedding_time, 3),
            "search_time": round(self.search_time, 3),
            "llm_time": round(self.llm_time, 3),
            "num_chunks": self.num_chunks,
            "context_length": self.context_length,
            "cache_hit": self.cache_hit
        }


# ============================================================================
# RAG 결과 데이터 클래스
# ============================================================================

@dataclass
class RAGResult:
    """RAG 파이프라인 실행 결과."""
    
    answer: str
    """생성된 답변"""
    
    sources: List[Dict[str, Any]]
    """참조한 문서 출처 정보"""
    
    context_used: str
    """실제 사용된 컨텍스트"""
    
    num_chunks: int
    """사용된 청크 개수"""
    
    metadata: Optional[Dict[str, Any]] = None
    """추가 메타데이터"""
    
    metrics: Optional[RAGMetrics] = None
    """성능 메트릭 (Phase 2)"""


# ============================================================================
# RAG 파이프라인
# ============================================================================

class RAGPipeline:
    """RAG 파이프라인 - 검색 증강 생성을 위한 통합 인터페이스.
    
    Phase 2 개선:
    - 임베딩 캐싱 (LRU)
    - 검색 결과 캐싱 (TTL)
    - 성능 메트릭 측정
    - 스트리밍 응답 지원
    """
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        retriever_type: str = "advanced",
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_cache: bool = True
    ):
        """
        RAG 파이프라인을 초기화합니다.
        
        Args:
            system_prompt: 시스템 프롬프트 (None이면 기본값 사용)
            user_prompt_template: 사용자 프롬프트 템플릿
            retriever_type: 검색기 타입 ("vector", "bm25", "hybrid", "advanced")
            top_k: 검색할 문서 개수 (None이면 config 사용)
            temperature: LLM 온도 파라미터
            max_tokens: 최대 생성 토큰 수
            enable_cache: 캐싱 활성화 여부 (Phase 2)
        """
        self.system_prompt = system_prompt or config.PROMPT.SYSTEM_PROMPT
        self.user_prompt_template = user_prompt_template or config.PROMPT.USER_PROMPT_TEMPLATE
        
        self.top_k = top_k or config.RAG.TOP_K
        self.temperature = temperature or config.RAG.TEMPERATURE
        self.max_tokens = max_tokens or config.RAG.MAX_TOKENS
        self.enable_cache = enable_cache
        
        # Ollama 클라이언트 초기화
        self.ollama_client = get_ollama_client()
        
        # 벡터 저장소 초기화
        self.vector_store = get_vector_store()
        
        # 검색기 초기화
        self.retriever = get_retriever(
            retriever_type=retriever_type,
            vector_store=self.vector_store
        )
        
        logger.info(f"✓ RAGPipeline 초기화됨 (retriever={retriever_type}, top_k={self.top_k}, cache={enable_cache})")
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        include_sources: bool = True,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_metrics: bool = False
    ) -> RAGResult:
        """
        질문에 대한 답변을 생성합니다. (Phase 2: 메트릭 측정 포함)
        
        Args:
            question: 사용자 질문
            top_k: 검색할 문서 개수 (None이면 기본값 사용)
            include_sources: 출처 정보 포함 여부
            filter_metadata: 검색 시 메타데이터 필터
            include_metrics: 성능 메트릭 포함 여부
            
        Returns:
            RAG 실행 결과
            
        Raises:
            ValueError: 질문이 비어있는 경우
            RuntimeError: RAG 파이프라인 실행 실패
        """
        if not question or not question.strip():
            raise ValueError("질문은 비어있을 수 없습니다")
        
        # 메트릭 초기화
        metrics = RAGMetrics() if include_metrics else None
        start_time = time.time()
        
        logger.info(f"RAG 쿼리 시작: '{question[:50]}...'")
        
        try:
            # 0. 쿼리 확장 (Query Expansion)
            # 원본 질문과 재작성된 쿼리를 모두 사용하여 검색 범위를 넓힘
            search_queries = [question]
            if config.RAG.ENABLE_QUERY_REWRITING:
                rewritten_query = self._rewrite_query(question)
                if rewritten_query != question:
                    logger.info(f"쿼리 재작성: '{question}' -> '{rewritten_query}'")
                    search_queries.append(rewritten_query)

            # 1. 다중 쿼리 검색 및 결과 병합
            k = top_k or self.top_k
            # 쿼리 확장 시 각 쿼리별로 더 많은 후보를 가져와서 병합 (Recall 향상)
            candidate_k = k * 3
            search_start = time.time()
            
            merged_results: Dict[str, SearchResult] = {}
            for query_text in search_queries:
                results = self._search_with_cache(query_text, candidate_k, filter_metadata)
                for result in results:
                    # 이미 있는 문서라면 더 높은 점수로 업데이트
                    if result.id not in merged_results or result.score > merged_results[result.id].score:
                        merged_results[result.id] = result
            
            # 점수순 정렬 후 상위 k개 선택
            search_results = sorted(merged_results.values(), key=lambda x: x.score, reverse=True)[:k]
            
            if metrics:
                metrics.search_time = time.time() - search_start
            
            if not search_results:
                logger.warning("검색 결과 없음")
                if metrics:
                    metrics.query_time = time.time() - start_time
                
                return RAGResult(
                    answer="관련 문서를 찾을 수 없습니다. 다른 질문을 시도해보세요.",
                    sources=[],
                    context_used="",
                    num_chunks=0,
                    metadata={"no_results": True},
                    metrics=metrics
                )
            
            logger.info(f"검색 완료: {len(search_results)}개 청크 발견")
            
            # 디버깅: 검색 결과 미리보기 (최대 15개)
            for i, result in enumerate(search_results[:15], 1):
                preview = result.content[:100].replace('\n', ' ')
                logger.info(f"  청크 {i}: score={result.score:.4f}, content={preview}...")
            
            # 2. 검색 결과 품질 필터링 (상대적 점수 기반)
            filtered_results = self._filter_low_quality_results(search_results)
            
            if not filtered_results:
                logger.warning("품질 필터링 후 결과 없음")
                if metrics:
                    metrics.query_time = time.time() - start_time
                
                return RAGResult(
                    answer="관련 문서를 찾을 수 없습니다. 다른 질문을 시도해보세요.",
                    sources=[],
                    context_used="",
                    num_chunks=0,
                    metadata={"filtered_out": True},
                    metrics=metrics
                )
            
            # 3. 컨텍스트 구성
            context = self._build_context(filtered_results)
            
            # 디버깅: 컨텍스트 미리보기
            context_preview = context[:300].replace('\n', ' ')
            logger.info(f"컨텍스트 생성됨 (길이: {len(context)}): {context_preview}...")
            
            if metrics:
                metrics.num_chunks = len(filtered_results)
                metrics.context_length = len(context)
            
            # 4. 프롬프트 생성
            prompt = self._build_prompt(question, context)
            
            # 5. LLM으로 답변 생성 (시간 측정)
            llm_start = time.time()
            answer = self._generate_answer(prompt)
            if metrics:
                metrics.llm_time = time.time() - llm_start
            
            # 6. 출처 정보 추출
            sources = []
            if include_sources:
                sources = self._extract_sources(filtered_results)
            
            # 전체 시간
            if metrics:
                metrics.query_time = time.time() - start_time
            
            logger.info(f"RAG 쿼리 완료 ({metrics.query_time:.2f}초)" if metrics else "RAG 쿼리 완료")
            
            return RAGResult(
                answer=answer,
                sources=sources,
                context_used=context,
                num_chunks=len(filtered_results),
                metadata={
                    "top_k": k,
                    "temperature": self.temperature
                },
                metrics=metrics
            )
        
        except Exception as e:
            logger.error(f"RAG 파이프라인 실행 실패: {str(e)}")
            raise RuntimeError(f"답변 생성 중 오류 발생: {str(e)}")
    
    def stream_query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Generator[str, None, RAGMetrics]:
        """
        질문에 대한 답변을 스트리밍 방식으로 생성합니다. (Phase 2)
        
        Args:
            question: 사용자 질문
            top_k: 검색할 문서 개수
            filter_metadata: 검색 시 메타데이터 필터
            
        Yields:
            생성된 답변 텍스트 청크
            
        Returns:
            성능 메트릭
            
        Raises:
            ValueError: 질문이 비어있는 경우
            RuntimeError: RAG 파이프라인 실행 실패
        """
        if not question or not question.strip():
            raise ValueError("질문은 비어있을 수 없습니다")
        
        metrics = RAGMetrics()
        start_time = time.time()
        
        logger.info(f"RAG 스트리밍 쿼리 시작: '{question[:50]}...'")
        
        try:
            # 1. 문서 검색
            k = top_k or self.top_k
            search_start = time.time()
            search_results = self._search_with_cache(question, k, filter_metadata)
            metrics.search_time = time.time() - search_start
            
            if not search_results:
                logger.warning("검색 결과 없음")
                yield "관련 문서를 찾을 수 없습니다. 다른 질문을 시도해보세요."
                metrics.query_time = time.time() - start_time
                return metrics
            
            # 2. 품질 필터링
            filtered_results = self._filter_low_quality_results(search_results)
            
            if not filtered_results:
                logger.warning("품질 필터링 후 결과 없음")
                yield "관련 문서를 찾을 수 없습니다. 다른 질문을 시도해보세요."
                metrics.query_time = time.time() - start_time
                return metrics
            
            # 3. 컨텍스트 구성
            context = self._build_context(filtered_results)
            metrics.num_chunks = len(filtered_results)
            metrics.context_length = len(context)
            
            # 4. 프롬프트 생성
            prompt = self._build_prompt(question, context)
            
            # 5. LLM 스트리밍 답변 생성
            llm_start = time.time()
            for chunk in self._stream_answer(prompt):
                yield chunk
            
            metrics.llm_time = time.time() - llm_start
            metrics.query_time = time.time() - start_time
            
            logger.info(f"RAG 스트리밍 쿼리 완료 ({metrics.query_time:.2f}초)")
            
            return metrics
        
        except Exception as e:
            logger.error(f"RAG 스트리밍 실패: {str(e)}")
            raise RuntimeError(f"스트리밍 답변 생성 중 오류 발생: {str(e)}")
    
    def _search_with_cache(
        self,
        question: str,
        k: int,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        질문에 대한 관련 문서를 검색합니다. (Phase 2: 캐싱 적용)
        
        Args:
            question: 검색 질문
            k: 반환할 결과 개수
            filter_metadata: 메타데이터 필터
            
        Returns:
            검색 결과 리스트
        """
        global _search_cache_hits, _search_cache_misses
        
        # 캐시 키 생성
        if self.enable_cache:
            filter_str = str(sorted(filter_metadata.items())) if filter_metadata else ""
            cache_key = hashlib.md5(
                f"{question}_{k}_{filter_str}".encode()
            ).hexdigest()
            
            # 캐시 확인
            if cache_key in _search_cache:
                _search_cache_hits += 1
                logger.debug(f"캐시 히트: {cache_key[:8]}")
                return _search_cache[cache_key]
            
            _search_cache_misses += 1
        
        # 캐시 미스 또는 캐시 비활성화 - 실제 검색 수행
        try:
            results = self.retriever.search(query=question, k=k)
            
            # 메타데이터 필터링
            if filter_metadata:
                results = [
                    r for r in results
                    if all(
                        r.metadata.get(key) == value
                        for key, value in filter_metadata.items()
                    )
                ]
            
            # 유사도 임계값 필터링은 제거 - 이미 retriever에서 top-k로 제한됨
            # ChromaDB의 거리 기반 점수는 정규화가 일관적이지 않으므로
            # 하드 threshold 대신 상위 k개 결과만 사용
            
            logger.info(f"검색 결과: {len(results)}개")
            
            # 캐시에 저장
            if self.enable_cache:
                _search_cache[cache_key] = results
            
            return results
        
        except Exception as e:
            logger.error(f"검색 실패: {str(e)}")
            return []
    
    def _rewrite_query(self, question: str) -> str:
        """
        검색 품질 향상을 위해 쿼리를 재작성합니다.
        
        Args:
            question: 원본 질문
            
        Returns:
            재작성된 검색 쿼리
        """
        try:
            prompt = config.PROMPT.QUERY_REWRITE_TEMPLATE.format(question=question)
            
            # 빠른 응답을 위해 낮은 temperature 사용
            response = self.ollama_client.generate(
                prompt=prompt,
                temperature=0.1,
                num_predict=50
            )
            
            cleaned_response = response.strip().strip('"')
            return cleaned_response
            
        except Exception as e:
            logger.warning(f"쿼리 재작성 실패: {e}")
            return question

    def _filter_low_quality_results(
        self,
        results: List[SearchResult],
        min_relative_score: float = 0.3
    ) -> List[SearchResult]:
        """
        낮은 품질의 검색 결과를 필터링합니다.
        
        Args:
            results: 검색 결과 리스트
            min_relative_score: 최고 점수 대비 최소 상대 점수 (0~1)
            
        Returns:
            필터링된 검색 결과
        """
        if not results:
            return []
        
        # 최고 점수 찾기
        max_score = max(r.score for r in results)
        
        if max_score <= 0:
            return results
        
        # 상대 점수 기반 필터링
        filtered = [
            r for r in results
            if r.score / max_score >= min_relative_score
        ]
        
        if len(filtered) < len(results):
            logger.info(f"품질 필터링: {len(results)}개 → {len(filtered)}개")
        
        return filtered
    
    def _stream_answer(self, prompt: str) -> Generator[str, None, None]:
        """
        프롬프트로부터 답변을 스트리밍 방식으로 생성합니다. (Phase 2)
        
        Args:
            prompt: 입력 프롬프트
            
        Yields:
            생성된 텍스트 청크
        """
        try:
            for chunk in self.ollama_client.stream_generate(
                prompt=prompt,
                temperature=self.temperature,
                num_predict=self.max_tokens
            ):
                yield chunk
        
        except Exception as e:
            logger.error(f"스트리밍 답변 생성 실패: {str(e)}")
            raise RuntimeError(f"LLM 스트리밍 생성 중 오류: {str(e)}")
    
    def _build_context(self, search_results: List[SearchResult]) -> str:
        """
        검색 결과로부터 컨텍스트를 구성합니다.
        섹션 제목이 있으면 우선적으로 표시합니다.
        "Lost in the Middle" 방지를 위해 중요 문서를 재배치합니다.
        
        Args:
            search_results: 검색 결과 리스트
            
        Returns:
            구성된 컨텍스트 문자열
        """
        if not search_results:
            return ""
        
        # Lost in the Middle 방지: [1, 2, 3, 4, 5] -> [1, 3, 5, 4, 2]
        # 가장 중요한(1) 문서가 맨 앞, 그 다음(2)이 맨 뒤로 배치되도록 재정렬
        # (이미 score 순으로 정렬되어 있다고 가정)
        reordered_results = []
        if len(search_results) > 2:
            # 홀수 번째는 앞에서부터, 짝수 번째는 뒤에서부터 채움 (대략적)
            # 정확한 Lost in the Middle reordering:
            # list[0], list[2], list[4], ..., list[3], list[1]
            # 혹은 간단하게: 1등 -> 맨 앞, 2등 -> 맨 뒤, 3등 -> 1등 뒤...
            # 여기서는 단순히 가장 좋은 1, 2등을 양 끝에 배치하는 전략 사용
            
            # deque를 사용하면 편함
            from collections import deque
            dq = deque(search_results)
            reordered = []
            while dq:
                if len(reordered) % 2 == 0:
                    reordered.append(dq.popleft())  # 상위를 앞쪽에
                else:
                    reordered.append(dq.popleft())  # 그 다음 상위를 (나중에 reverse할 뒤쪽에 배치가 아니라, 순서대로 넣는데...)
            
            # LangChain 스타일 Reorder:
            # [1, 2, 3, 4, 5] -> [1, 3, 5, 4, 2]
            # 1 (0) -> index 0
            # 2 (1) -> index 4
            # 3 (2) -> index 1
            # 4 (3) -> index 3
            # 5 (4) -> index 2
            
            # 구현:
            new_order = [None] * len(search_results)
            left = 0
            right = len(search_results) - 1
            for i, item in enumerate(search_results):
                if i % 2 == 0:
                    new_order[left] = item
                    left += 1
                else:
                    new_order[right] = item
                    right -= 1
            reordered_results = new_order
        else:
            reordered_results = search_results

        context_parts = []
        
        for i, result in enumerate(reordered_results, 1):
            # 섹션 제목 우선 사용 (청크 품질 향상)
            header = ""
            if result.metadata:
                section_title = result.metadata.get("section_title", "")
                source_name = result.metadata.get("source", "")
                
                # 원래 순위(Rank) 정보도 포함하면 좋음
                # result 객체에 원래 rank 정보가 없으므로 생략하거나, search_results에서 index 찾기
                original_rank = search_results.index(result) + 1
                
                if section_title:
                    # 섹션 제목이 있으면 컨텍스트 헤더로 사용
                    header = f"[Doc {i} / Rank {original_rank}] {section_title}"
                    if source_name:
                        # 파일명만 추출 (경로 제거)
                        from pathlib import Path
                        filename = Path(source_name).name
                        header += f" (from {filename})"
                elif source_name:
                    from pathlib import Path
                    filename = Path(source_name).name
                    header = f"[Doc {i} / Rank {original_rank}] (Source: {filename})"
                else:
                    header = f"[Doc {i} / Rank {original_rank}]"
            else:
                header = f"[Doc {i}]"
            
            # 컨텍스트 구성
            context_parts.append(
                f"{header}\n{result.content.strip()}"
            )
        
        # 문서들 사이에 빈 줄로 구분
        context = "\n\n".join(context_parts)
        
        # 최대 길이 확인 (간단한 문자 기반 체크)
        max_length = config.RAG.MAX_CONTEXT_LENGTH
        if len(context) > max_length:
            context = context[:max_length] + "\n...(생략)..."
            logger.warning(f"컨텍스트가 최대 길이를 초과하여 잘림 ({max_length} 문자)")
        
        return context
    
    def _build_prompt(self, question: str, context: str) -> str:
        """
        질문과 컨텍스트로부터 프롬프트를 생성합니다.
        
        Args:
            question: 사용자 질문
            context: 검색된 컨텍스트
            
        Returns:
            생성된 프롬프트
        """
        # 사용자 프롬프트 생성
        user_prompt = self.user_prompt_template.format(
            context=context,
            question=question
        )
        
        # 시스템 프롬프트와 결합
        full_prompt = f"{self.system_prompt}\n\n{user_prompt}"
        
        return full_prompt
    
    def _generate_answer(self, prompt: str) -> str:
        """
        프롬프트로부터 답변을 생성합니다.
        
        Args:
            prompt: 입력 프롬프트
            
        Returns:
            생성된 답변
        """
        try:
            # 디버깅: 프롬프트 전체 로그 출력
            logger.info(f"LLM 전체 프롬프트:\n{prompt}")
            
            # 디버깅: 프롬프트 미리보기 (기존 로그)
            prompt_preview = prompt[:500].replace('\n', ' ')
            logger.info(f"LLM 프롬프트 전송 중 (길이: {len(prompt)}): {prompt_preview}...")
            
            answer = self.ollama_client.generate(
                prompt=prompt,
                temperature=self.temperature,
                num_predict=self.max_tokens
            )
            
            # 디버깅: 답변 미리보기
            answer_preview = answer[:200].replace('\n', ' ')
            logger.info(f"LLM 답변 수신됨 (길이: {len(answer)}): {answer_preview}...")
            
            return answer.strip()
        
        except Exception as e:
            logger.error(f"답변 생성 실패: {str(e)}")
            raise RuntimeError(f"LLM 답변 생성 중 오류: {str(e)}")
    
    def _extract_sources(self, search_results: List[SearchResult]) -> List[Dict[str, Any]]:
        """
        검색 결과로부터 출처 정보를 추출합니다.
        
        Args:
            search_results: 검색 결과 리스트
            
        Returns:
            출처 정보 리스트
        """
        sources = []
        
        for i, result in enumerate(search_results, 1):
            source_info = {
                "rank": i,
                "score": round(result.score, 4),
                "content": result.content,
                "content_preview": result.content[:200] + "..." if len(result.content) > 200 else result.content
            }
            
            # 메타데이터 추가
            if result.metadata:
                source_info.update({
                    "source": result.metadata.get("source", "알 수 없음"),
                    "chunk_id": result.metadata.get("chunk_id"),
                    "total_chunks": result.metadata.get("total_chunks")
                })
            
            sources.append(source_info)
        
        # 디버깅: 첫 번째 출처 확인
        if sources:
            logger.info(f"_extract_sources - 첫 출처 content 길이: {len(sources[0].get('content', ''))}")
        
        return sources
    
    def chat(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None
    ) -> RAGResult:
        """
        대화 히스토리를 고려한 질의응답.
        
        Args:
            question: 사용자 질문
            conversation_history: 이전 대화 기록 [{"role": "user/assistant", "content": "..."}]
            top_k: 검색할 문서 개수
            
        Returns:
            RAG 실행 결과
        """
        # 대화 히스토리를 컨텍스트에 포함
        if conversation_history:
            # 히스토리를 프롬프트에 추가
            history_text = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in conversation_history[-3:]  # 최근 3개만
            ])
            
            enhanced_question = f"이전 대화:\n{history_text}\n\n현재 질문: {question}"
            return self.query(question=enhanced_question, top_k=top_k)
        else:
            return self.query(question=question, top_k=top_k)
    
    @staticmethod
    def get_cache_stats() -> Dict[str, Any]:
        """
        캐시 통계를 반환합니다. (Phase 2)
        
        Returns:
            캐시 히트/미스 통계
        """
        global _search_cache_hits, _search_cache_misses
        
        total = _search_cache_hits + _search_cache_misses
        hit_rate = (_search_cache_hits / total * 100) if total > 0 else 0
        
        return {
            "cache_hits": _search_cache_hits,
            "cache_misses": _search_cache_misses,
            "total_requests": total,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_size": len(_search_cache),
            "cache_maxsize": _search_cache.maxsize
        }
    
    @staticmethod
    def clear_cache():
        """캐시를 초기화합니다. (Phase 2)"""
        global _search_cache_hits, _search_cache_misses
        
        _search_cache.clear()
        _search_cache_hits = 0
        _search_cache_misses = 0
        logger.info("캐시가 초기화되었습니다.")


# ============================================================================
# 팩토리 함수
# ============================================================================

_rag_pipeline_instance = None


def get_rag_pipeline(
    retriever_type: str = "advanced",
    **kwargs
) -> RAGPipeline:
    """
    RAG 파이프라인 싱글톤 인스턴스를 반환합니다.
    
    Args:
        retriever_type: 검색기 타입
        **kwargs: RAGPipeline 추가 파라미터
        
    Returns:
        RAG 파이프라인 인스턴스
    """
    global _rag_pipeline_instance
    
    if _rag_pipeline_instance is None:
        _rag_pipeline_instance = RAGPipeline(
            retriever_type=retriever_type,
            **kwargs
        )
    
    return _rag_pipeline_instance


def create_rag_pipeline(
    retriever_type: str = "advanced",
    **kwargs
) -> RAGPipeline:
    """
    새로운 RAG 파이프라인 인스턴스를 생성합니다.
    
    Args:
        retriever_type: 검색기 타입
        **kwargs: RAGPipeline 추가 파라미터
        
    Returns:
        새로운 RAG 파이프라인 인스턴스
    """
    return RAGPipelin