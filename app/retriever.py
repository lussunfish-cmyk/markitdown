"""
문서 검색 로직 구현.
벡터 검색, 키워드 검색, 하이브리드 검색, 리랭킹 기능 제공.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict
import re
import string

import numpy as np # type: ignore
from rank_bm25 import BM25Okapi  # type: ignore
try:
    from sentence_transformers import CrossEncoder # type: ignore
except ImportError:
    CrossEncoder = None

from .config import config
from .llm_client import get_llm_client
from .vector_store import get_vector_store, VectorStore

logger = logging.getLogger(__name__)


# ============================================================================
# 검색 결과 타입
# ============================================================================

class SearchResult:
    """검색 결과를 나타내는 클래스."""
    
    def __init__(
        self,
        id: str,
        content: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        검색 결과를 초기화합니다.
        
        Args:
            id: 문서 청크 ID
            content: 문서 내용
            score: 유사도 점수 (높을수록 유사)
            metadata: 메타데이터
        """
        self.id = id
        self.content = content
        self.score = score
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환합니다."""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata
        }
    
    def __repr__(self) -> str:
        return f"SearchResult(id={self.id}, score={self.score:.4f})"


# ============================================================================
# 기본 검색기 추상 클래스
# ============================================================================

class BaseRetriever(ABC):
    """검색기 기본 인터페이스."""
    
    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """
        쿼리에 대한 검색을 수행합니다.
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 개수
            
        Returns:
            검색 결과 리스트
        """
        pass


# ============================================================================
# 벡터 검색기
# ============================================================================

class VectorRetriever(BaseRetriever):
    """벡터 유사도 기반 검색기."""
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        벡터 검색기를 초기화합니다.
        
        Args:
            vector_store: 벡터 저장소 (None이면 기본 저장소 사용)
        """
        self.vector_store = vector_store or get_vector_store()
        self.llm_client = get_llm_client()
        logger.info("✓ VectorRetriever 초기화됨")
    
    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """
        벡터 유사도 검색을 수행합니다.
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 개수
            
        Returns:
            검색 결과 리스트 (유사도 높은 순)
        """
        # 쿼리 임베딩 생성
        query_embedding = self.llm_client.embed(query)
        
        # 벡터 검색
        raw_results = self.vector_store.search(query_embedding, k=k)
        
        # SearchResult로 변환
        results = []
        for result in raw_results:
            search_result = SearchResult(
                id=result.get("id", ""),
                content=result.get("document", ""),
                score=result.get("score", 0.0),
                metadata=result.get("metadata", {})
            )
            results.append(search_result)
        
        logger.info(f"벡터 검색 완료: {len(results)}개 결과")
        return results


# ============================================================================
# BM25 키워드 검색기
# ============================================================================

class BM25Retriever(BaseRetriever):
    """BM25 알고리즘 기반 키워드 검색기."""
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        BM25 검색기를 초기화합니다.
        
        Args:
            vector_store: 벡터 저장소 (문서 가져오기용)
        """
        self.vector_store = vector_store or get_vector_store()
        self.bm25_index = None
        self.documents = []
        self.doc_ids = []
        self._build_index()
        logger.info("✓ BM25Retriever 초기화됨")
    
    def _tokenize(self, text: str) -> List[str]:
        """텍스트를 토큰화합니다 (구두점 제거)."""
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        return text.split()

    def _build_index(self) -> None:
        """BM25 인덱스를 구축합니다."""
        try:
            # 벡터 저장소에서 모든 문서 가져오기
            count = self.vector_store.count()
            if count == 0:
                logger.warning("벡터 저장소에 문서가 없습니다")
                return
            
            # 모든 문서 조회 (ChromaDB의 경우 peek 사용)
            if hasattr(self.vector_store, 'collection'):
                data = self.vector_store.collection.get()
                self.doc_ids = data.get('ids', [])
                self.documents = data.get('documents', [])
            else:
                logger.warning("BM25 인덱스 구축 실패: 문서 조회 불가")
                return
            
            if not self.documents:
                logger.warning("문서를 가져올 수 없습니다")
                return
            
            # 토큰화 (간단한 공백 분리)
            tokenized_docs = [self._tokenize(doc) for doc in self.documents]
            
            # BM25 인덱스 생성
            self.bm25_index = BM25Okapi(tokenized_docs)
            logger.info(f"BM25 인덱스 구축 완료: {len(self.documents)}개 문서")
            
        except Exception as e:
            logger.error(f"BM25 인덱스 구축 실패: {str(e)}")
            raise
    
    def rebuild_index(self) -> None:
        """인덱스를 재구축합니다."""
        self._build_index()
    
    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """
        BM25 키워드 검색을 수행합니다.
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 개수
            
        Returns:
            검색 결과 리스트 (BM25 점수 높은 순)
        """
        if self.bm25_index is None or not self.documents:
            logger.warning("BM25 인덱스가 없습니다")
            return []
        
        # 쿼리 토큰화
        tokenized_query = self._tokenize(query)
        
        # BM25 점수 계산
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # 점수 높은 순으로 정렬
        top_indices = np.argsort(scores)[::-1][:k]
        
        # SearchResult로 변환
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 점수가 0보다 큰 것만
                search_result = SearchResult(
                    id=self.doc_ids[idx],
                    content=self.documents[idx],
                    score=float(scores[idx]),
                    metadata={}
                )
                results.append(search_result)
        
        logger.info(f"BM25 검색 완료: {len(results)}개 결과")
        return results


# ============================================================================
# 하이브리드 검색기
# ============================================================================

class HybridRetriever(BaseRetriever):
    """벡터 검색과 BM25 검색을 결합한 하이브리드 검색기."""
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        alpha: Optional[float] = None,
        fusion_method: str = "weighted"
    ):
        """
        하이브리드 검색기를 초기화합니다.
        
        Args:
            vector_store: 벡터 저장소
            alpha: 벡터 검색 가중치 (0~1, 1-alpha는 BM25 가중치, None이면 config 사용)
            fusion_method: 결합 방식 ("weighted" 또는 "rrf")
        """
        self.vector_retriever = VectorRetriever(vector_store)
        self.bm25_retriever = BM25Retriever(vector_store)
        self.alpha = alpha if alpha is not None else config.RETRIEVER.HYBRID_ALPHA
        self.fusion_method = fusion_method
        
        if fusion_method not in ["weighted", "rrf"]:
            raise ValueError(f"지원하지 않는 fusion_method: {fusion_method}")
        
        logger.info(f"✓ HybridRetriever 초기화됨 (alpha={self.alpha}, method={fusion_method})")
    
    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """
        하이브리드 검색을 수행합니다.
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 개수
            
        Returns:
            검색 결과 리스트 (결합 점수 높은 순)
        """
        # 각각 더 많은 결과 검색 (후보 풀 확보)
        candidate_k = k * 3
        
        # 벡터 검색
        vector_results = self.vector_retriever.search(query, k=candidate_k)
        
        # BM25 검색
        bm25_results = self.bm25_retriever.search(query, k=candidate_k)
        
        # 결과 결합
        if self.fusion_method == "weighted":
            combined_results = self._weighted_fusion(vector_results, bm25_results)
        else:  # rrf
            combined_results = self._rrf_fusion(vector_results, bm25_results)
        
        # 상위 k개 반환
        final_results = combined_results[:k]
        
        logger.info(f"하이브리드 검색 완료: {len(final_results)}개 결과")
        return final_results
    
    def _weighted_fusion(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        가중 점수 방식으로 결과를 결합합니다.
        
        Args:
            vector_results: 벡터 검색 결과
            bm25_results: BM25 검색 결과
            
        Returns:
            결합된 검색 결과
        """
        # 점수 정규화를 위한 최대값 계산
        vector_max = max([r.score for r in vector_results], default=1.0)
        bm25_max = max([r.score for r in bm25_results], default=1.0)
        
        # 문서별 점수 합산
        combined_scores: Dict[str, Tuple[float, str, Dict[str, Any]]] = {}
        
        # 벡터 검색 점수 추가
        for result in vector_results:
            normalized_score = result.score / vector_max if vector_max > 0 else 0
            weighted_score = self.alpha * normalized_score
            combined_scores[result.id] = (
                weighted_score,
                result.content,
                result.metadata
            )
        
        # BM25 검색 점수 추가
        for result in bm25_results:
            normalized_score = result.score / bm25_max if bm25_max > 0 else 0
            weighted_score = (1 - self.alpha) * normalized_score
            
            if result.id in combined_scores:
                # 기존 점수에 추가
                old_score, content, metadata = combined_scores[result.id]
                combined_scores[result.id] = (
                    old_score + weighted_score,
                    content,
                    metadata
                )
            else:
                # 새로운 항목
                combined_scores[result.id] = (
                    weighted_score,
                    result.content,
                    result.metadata
                )
        
        # SearchResult로 변환 및 정렬
        results = [
            SearchResult(id=doc_id, content=content, score=score, metadata=metadata)
            for doc_id, (score, content, metadata) in combined_scores.items()
        ]
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def _rrf_fusion(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
        k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion (RRF) 방식으로 결과를 결합합니다.
        
        Args:
            vector_results: 벡터 검색 결과
            bm25_results: BM25 검색 결과
            k: RRF 파라미터 (None이면 config 사용)
            
        Returns:
            결합된 검색 결과
        """
        if k is None:
            k = config.RETRIEVER.RRF_K
        
        # 문서별 RRF 점수 계산
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_info: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        
        # 벡터 검색 순위 반영
        for rank, result in enumerate(vector_results):
            rrf_scores[result.id] += 1.0 / (k + rank + 1)
            doc_info[result.id] = (result.content, result.metadata)
        
        # BM25 검색 순위 반영
        for rank, result in enumerate(bm25_results):
            rrf_scores[result.id] += 1.0 / (k + rank + 1)
            if result.id not in doc_info:
                doc_info[result.id] = (result.content, result.metadata)
        
        # SearchResult로 변환 및 정렬
        results = [
            SearchResult(
                id=doc_id,
                content=doc_info[doc_id][0],
                score=score,
                metadata=doc_info[doc_id][1]
            )
            for doc_id, score in rrf_scores.items()
        ]
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results


# ============================================================================
# 리랭커
# ============================================================================

class Reranker:
    """검색 결과를 재정렬하는 리랭커."""
    
    def __init__(self, method: str = "cross_encoder"):
        """
        리랭커를 초기화합니다.
        
        Args:
            method: 리랭킹 방법 ("bm25", "score", "cross_encoder")
        """
        self.method = method
        self.cross_encoder = None
        
        if self.method == "cross_encoder":
            if CrossEncoder is None:
                logger.warning("sentence-transformers가 설치되지 않아 bm25로 대체합니다.")
                self.method = "bm25"
            else:
                try:
                    logger.info(f"CrossEncoder 모델 로딩 중: {config.RETRIEVER.RERANKER_MODEL}")
                    self.cross_encoder = CrossEncoder(config.RETRIEVER.RERANKER_MODEL)
                except Exception as e:
                    logger.error(f"CrossEncoder 로딩 실패: {e}. bm25로 대체합니다.")
                    self.method = "bm25"
                
        logger.info(f"✓ Reranker 초기화됨 (method={self.method})")
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        검색 결과를 재정렬합니다.
        
        Args:
            query: 원본 쿼리
            results: 검색 결과 리스트
            top_k: 반환할 결과 개수 (None이면 전체)
            
        Returns:
            재정렬된 검색 결과
        """
        if not results:
            return []
        
        if self.method == "cross_encoder" and self.cross_encoder:
            reranked = self._rerank_with_cross_encoder(query, results)
        elif self.method == "bm25":
            reranked = self._rerank_with_bm25(query, results)
        elif self.method == "score":
            # 이미 점수로 정렬되어 있으므로 그대로 반환
            reranked = results
        else:
            logger.warning(f"지원하지 않는 리랭킹 방법: {self.method}")
            reranked = results
        
        # top_k 적용
        if top_k is not None:
            reranked = reranked[:top_k]
        
        logger.info(f"리랭킹 완료: {len(results)}개 -> {len(reranked)}개")
        return reranked
    
    def _rerank_with_cross_encoder(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """Cross-Encoder로 정밀하게 재정렬합니다."""
        if not results:
            return []
            
        # (쿼리, 문서) 쌍 생성
        pairs = [[query, r.content] for r in results]
        
        try:
            # 메모리 확보를 위해 캐시 정리
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            # 점수 계산
            # OOM 방지를 위해 배치 사이즈 대폭 제한 (12 -> 4)
            scores = self.cross_encoder.predict(
                pairs, 
                batch_size=4,
                show_progress_bar=False
            )
            
            # 결과 업데이트
            reranked_results = []
            for i, result in enumerate(results):
                # 기존 메타데이터 유지하면서 점수만 업데이트
                new_result = SearchResult(
                    id=result.id,
                    content=result.content,
                    score=float(scores[i]),
                    metadata=result.metadata
                )
                reranked_results.append(new_result)
            
            # 점수 내림차순 정렬
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            return reranked_results
            
        except Exception as e:
            logger.error(f"Cross-Encoder 리랭킹 실패: {e}")
            return results

    def _tokenize(self, text: str) -> List[str]:
        """텍스트를 토큰화합니다 (구두점 제거)."""
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        return text.split()

    def _rerank_with_bm25(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        BM25 점수로 재정렬합니다.
        
        Args:
            query: 검색 쿼리
            results: 검색 결과
            
        Returns:
            재정렬된 결과
        """
        # 문서 리스트 준비
        documents = [result.content for result in results]
        
        # 토큰화
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        tokenized_query = self._tokenize(query)
        
        # BM25 인덱스 생성 및 점수 계산
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(tokenized_query)
        
        # 새로운 점수로 SearchResult 업데이트
        reranked_results = []
        for i, result in enumerate(results):
            new_result = SearchResult(
                id=result.id,
                content=result.content,
                score=float(scores[i]),
                metadata=result.metadata
            )
            reranked_results.append(new_result)
        
        # 점수로 정렬
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        return reranked_results


# ============================================================================
# 고급 검색기 (하이브리드 + 리랭킹)
# ============================================================================

class AdvancedRetriever(BaseRetriever):
    """하이브리드 검색과 리랭킹을 결합한 고급 검색기."""
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        alpha: Optional[float] = None,
        fusion_method: str = "weighted",
        rerank_method: str = "bm25",
        use_rerank: bool = True
    ):
        """
        고급 검색기를 초기화합니다.
        
        Args:
            vector_store: 벡터 저장소
            alpha: 하이브리드 검색 가중치 (None이면 config 사용)
            fusion_method: 결합 방식 ("weighted" 또는 "rrf")
            rerank_method: 리랭킹 방법
            use_rerank: 리랭킹 사용 여부
        """
        self.hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            alpha=alpha,
            fusion_method=fusion_method
        )
        self.reranker = Reranker(method=rerank_method)
        self.use_rerank = use_rerank
        
        logger.info(f"✓ AdvancedRetriever 초기화됨 (rerank={use_rerank})")
    
    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """
        고급 검색을 수행합니다.
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 개수
            
        Returns:
            검색 결과 리스트
        """
        # 1. 하이브리드 검색으로 후보 추출
        if self.use_rerank:
            # 리랭킹을 사용하는 경우 더 많은 후보 확보
            candidates = self.hybrid_retriever.search(query, k=k*10)
        else:
            candidates = self.hybrid_retriever.search(query, k=k)
        
        # 2. 리랭킹 (선택)
        if self.use_rerank and candidates:
            final_results = self.reranker.rerank(query, candidates, top_k=k)
        else:
            final_results = candidates[:k]
        
        logger.info(f"고급 검색 완료: {len(final_results)}개 결과")
        return final_results


# ============================================================================
# 팩토리 함수
# ============================================================================

def create_retriever(
    retriever_type: str = "hybrid",
    vector_store: Optional[VectorStore] = None,
    **kwargs
) -> BaseRetriever:
    """
    검색기를 생성합니다.
    
    Args:
        retriever_type: 검색기 타입
            - "vector": 벡터 검색만
            - "bm25": BM25 검색만
            - "hybrid": 하이브리드 검색
            - "advanced": 하이브리드 + 리랭킹
        vector_store: 벡터 저장소
        **kwargs: 추가 파라미터
        
    Returns:
        검색기 인스턴스
    """
    if retriever_type == "vector":
        return VectorRetriever(vector_store=vector_store)
    
    elif retriever_type == "bm25":
        return BM25Retriever(vector_store=vector_store)
    
    elif retriever_type == "hybrid":
        return HybridRetriever(
            vector_store=vector_store,
            alpha=kwargs.get("alpha", None),  # None이면 config 사용
            fusion_method=kwargs.get("fusion_method", "weighted")
        )
    
    elif retriever_type == "advanced":
        return AdvancedRetriever(
            vector_store=vector_store,
            alpha=kwargs.get("alpha", None),  # None이면 config 사용
            fusion_method=kwargs.get("fusion_method", "weighted"),
            rerank_method=kwargs.get("rerank_method", "cross_encoder"),
            use_rerank=kwargs.get("use_rerank", config.RETRIEVER.USE_RERANKER)
        )
    
    else:
        raise ValueError(f"지원하지 않는 retriever_type: {retriever_type}")


def get_retriever(retriever_type: str = "advanced", **kwargs) -> BaseRetriever:
    """
    기본 검색기를 반환합니다.
    
    Args:
        retriever_type: 검색기 타입
        **kwargs: 추가 파라미터
        
    Returns:
        검색기 인스턴스
    """
    return create_retriever(retriever_type=retriever_type, **kwargs)
