"""
RAG (검색 증강 생성) 파이프라인 구현.
질의응답을 위한 검색, 컨텍스트 구성, 답변 생성 통합 모듈.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .config import config
from .ollama_client import get_ollama_client
from .vector_store import get_vector_store
from .retriever import get_retriever, SearchResult
from .schemas import RAGRequest, RAGResponse

logger = logging.getLogger(__name__)


# ============================================================================
# 프롬프트 템플릿
# ============================================================================

DEFAULT_SYSTEM_PROMPT = """당신은 전문적인 기술 문서 도우미입니다.
주어진 컨텍스트를 기반으로 사용자의 질문에 정확하고 명확하게 답변하세요.

지침:
1. 제공된 컨텍스트만을 사용하여 답변하세요.
2. 컨텍스트에 정보가 없으면 "제공된 문서에서 관련 정보를 찾을 수 없습니다"라고 답하세요.
3. 추측하거나 컨텍스트 외부의 지식을 사용하지 마세요.
4. 답변은 간결하고 명확하게 작성하세요.
5. 가능하면 컨텍스트에서 인용한 부분을 표시하세요."""

DEFAULT_USER_PROMPT_TEMPLATE = """컨텍스트:
{context}

질문: {question}

답변:"""


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


# ============================================================================
# RAG 파이프라인
# ============================================================================

class RAGPipeline:
    """RAG 파이프라인 - 검색 증강 생성을 위한 통합 인터페이스."""
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        retriever_type: str = "advanced",
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
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
        """
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.user_prompt_template = user_prompt_template or DEFAULT_USER_PROMPT_TEMPLATE
        
        self.top_k = top_k or config.RAG.TOP_K
        self.temperature = temperature or config.RAG.TEMPERATURE
        self.max_tokens = max_tokens or config.RAG.MAX_TOKENS
        
        # Ollama 클라이언트 초기화
        self.ollama_client = get_ollama_client()
        
        # 벡터 저장소 초기화
        self.vector_store = get_vector_store()
        
        # 검색기 초기화
        self.retriever = get_retriever(
            retriever_type=retriever_type,
            vector_store=self.vector_store
        )
        
        logger.info(f"✓ RAGPipeline 초기화됨 (retriever={retriever_type}, top_k={self.top_k})")
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        include_sources: bool = True,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> RAGResult:
        """
        질문에 대한 답변을 생성합니다.
        
        Args:
            question: 사용자 질문
            top_k: 검색할 문서 개수 (None이면 기본값 사용)
            include_sources: 출처 정보 포함 여부
            filter_metadata: 검색 시 메타데이터 필터
            
        Returns:
            RAG 실행 결과
            
        Raises:
            ValueError: 질문이 비어있는 경우
            RuntimeError: RAG 파이프라인 실행 실패
        """
        if not question or not question.strip():
            raise ValueError("질문은 비어있을 수 없습니다")
        
        logger.info(f"RAG 쿼리 시작: '{question[:50]}...'")
        
        try:
            # 1. 문서 검색
            k = top_k or self.top_k
            search_results = self._search(question, k, filter_metadata)
            
            if not search_results:
                logger.warning("검색 결과 없음")
                return RAGResult(
                    answer="관련 문서를 찾을 수 없습니다. 다른 질문을 시도해보세요.",
                    sources=[],
                    context_used="",
                    num_chunks=0,
                    metadata={"no_results": True}
                )
            
            logger.info(f"검색 완료: {len(search_results)}개 청크 발견")
            
            # 2. 컨텍스트 구성
            context = self._build_context(search_results)
            
            # 3. 프롬프트 생성
            prompt = self._build_prompt(question, context)
            
            # 4. LLM으로 답변 생성
            answer = self._generate_answer(prompt)
            
            # 5. 출처 정보 추출
            sources = []
            if include_sources:
                sources = self._extract_sources(search_results)
            
            logger.info("RAG 쿼리 완료")
            
            return RAGResult(
                answer=answer,
                sources=sources,
                context_used=context,
                num_chunks=len(search_results),
                metadata={
                    "top_k": k,
                    "temperature": self.temperature
                }
            )
        
        except Exception as e:
            logger.error(f"RAG 파이프라인 실행 실패: {str(e)}")
            raise RuntimeError(f"답변 생성 중 오류 발생: {str(e)}")
    
    def _search(
        self,
        question: str,
        k: int,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        질문에 대한 관련 문서를 검색합니다.
        
        Args:
            question: 검색 질문
            k: 반환할 결과 개수
            filter_metadata: 메타데이터 필터
            
        Returns:
            검색 결과 리스트
        """
        try:
            # 검색 실행
            results = self.retriever.search(query=question, k=k)
            
            # 메타데이터 필터링 (선택)
            if filter_metadata:
                results = [
                    r for r in results
                    if all(
                        r.metadata.get(key) == value
                        for key, value in filter_metadata.items()
                    )
                ]
            
            # 유사도 임계값 필터링
            threshold = config.RAG.SIMILARITY_THRESHOLD
            results = [r for r in results if r.score >= threshold]
            
            return results
        
        except Exception as e:
            logger.error(f"검색 실패: {str(e)}")
            return []
    
    def _build_context(self, search_results: List[SearchResult]) -> str:
        """
        검색 결과로부터 컨텍스트를 구성합니다.
        
        Args:
            search_results: 검색 결과 리스트
            
        Returns:
            구성된 컨텍스트 문자열
        """
        if not search_results:
            return ""
        
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            # 출처 정보 추가
            source_info = ""
            if result.metadata:
                source = result.metadata.get("source", "알 수 없음")
                source_info = f"[출처: {source}]"
            
            # 청크 내용
            context_parts.append(
                f"--- 문서 {i} {source_info} ---\n{result.content}\n"
            )
        
        context = "\n".join(context_parts)
        
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
            answer = self.ollama_client.generate(
                prompt=prompt,
                temperature=self.temperature,
                num_predict=self.max_tokens
            )
            
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
    return RAGPipeline(retriever_type=retriever_type, **kwargs)
