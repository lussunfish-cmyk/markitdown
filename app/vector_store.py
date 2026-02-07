"""
벡터 저장소 관리 모듈.
ChromaDB 및 FAISS 벡터 저장소를 위한 추상화 레이어.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path

import chromadb
from chromadb.config import Settings

from .config import config
from .schemas import DocumentChunk

logger = logging.getLogger(__name__)


# ============================================================================
# 추상 클래스
# ============================================================================

class VectorStore(ABC):
    """벡터 저장소 추상 인터페이스."""
    
    @abstractmethod
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        벡터와 문서를 저장소에 추가합니다.
        
        Args:
            ids: 문서 청크 ID 목록
            embeddings: 임베딩 벡터 목록
            documents: 문서 텍스트 목록
            metadatas: 메타데이터 딕셔너리 목록
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        유사도 검색을 수행합니다.
        
        Args:
            query_embedding: 쿼리 임베딩 벡터
            k: 반환할 결과 개수
            filter: 메타데이터 필터 (선택)
            
        Returns:
            검색 결과 목록. 각 결과는 다음을 포함:
            - id: 문서 청크 ID
            - document: 문서 텍스트
            - metadata: 메타데이터
            - distance: 거리 (낮을수록 유사)
            - score: 유사도 점수 (높을수록 유사)
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """
        문서를 저장소에서 삭제합니다.
        
        Args:
            ids: 삭제할 문서 청크 ID 목록
        """
        pass
    
    @abstractmethod
    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        ID로 문서를 조회합니다.
        
        Args:
            ids: 조회할 문서 청크 ID 목록
            
        Returns:
            문서 정보 목록
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """
        저장소의 총 문서 개수를 반환합니다.
        
        Returns:
            문서 개수
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """저장소의 모든 데이터를 삭제합니다."""
        pass
    
    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """
        컬렉션 정보를 반환합니다.
        
        Returns:
            컬렉션 통계 및 메타데이터
        """
        pass


# ============================================================================
# ChromaDB 구현
# ============================================================================

class ChromaVectorStore(VectorStore):
    """ChromaDB 기반 벡터 저장소."""
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[Path] = None
    ):
        """
        ChromaDB 벡터 저장소를 초기화합니다.
        
        Args:
            collection_name: 컬렉션 이름
            persist_directory: 영속성 디렉토리
        """
        self.collection_name = collection_name or config.VECTOR_STORE.CHROMA_COLLECTION_NAME
        self.persist_directory = persist_directory or config.VECTOR_STORE.PERSIST_DIR
        
        # 디렉토리 생성
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 컬렉션 가져오기 또는 생성
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"✓ 기존 컬렉션 로드됨: {self.collection_name} (문서 수: {self.collection.count()})")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # 코사인 유사도 사용
            )
            logger.info(f"✓ 새 컬렉션 생성됨: {self.collection_name}")
    
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        벡터와 문서를 ChromaDB에 추가합니다.
        
        Args:
            ids: 문서 청크 ID 목록
            embeddings: 임베딩 벡터 목록
            documents: 문서 텍스트 목록
            metadatas: 메타데이터 딕셔너리 목록
        """
        if not ids or not embeddings or not documents:
            raise ValueError("ids, embeddings, documents는 비어있을 수 없습니다")
        
        if len(ids) != len(embeddings) != len(documents):
            raise ValueError("ids, embeddings, documents의 길이가 일치해야 합니다")
        
        try:
            # ChromaDB에 추가 (중복 시 업데이트)
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"✓ {len(ids)}개 문서 청크 추가됨")
        except Exception as e:
            logger.error(f"✗ 문서 추가 실패: {str(e)}")
            raise RuntimeError(f"ChromaDB 추가 실패: {str(e)}")
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        ChromaDB에서 유사도 검색을 수행합니다.
        
        Args:
            query_embedding: 쿼리 임베딩 벡터
            k: 반환할 결과 개수
            filter: 메타데이터 필터 (where 조건)
            
        Returns:
            검색 결과 목록
        """
        if not query_embedding:
            raise ValueError("query_embedding은 비어있을 수 없습니다")
        
        try:
            # ChromaDB 검색
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self.collection.count()),
                where=filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # 결과 포맷팅
            formatted_results = []
            if results['ids'] is not None and len(results['ids']) > 0 and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    distance = results['distances'][0][i]
                    # 코사인 거리를 유사도로 변환: similarity = 1 - distance
                    similarity = 1.0 - distance
                    
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] is not None else {},
                        'distance': distance,
                        'score': similarity
                    })
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"✗ 검색 실패: {str(e)}")
            raise RuntimeError(f"ChromaDB 검색 실패: {str(e)}")
    
    def delete(self, ids: List[str]) -> None:
        """
        ChromaDB에서 문서를 삭제합니다.
        
        Args:
            ids: 삭제할 문서 청크 ID 목록
        """
        if not ids:
            logger.warning("삭제할 ID가 없습니다")
            return
        
        try:
            self.collection.delete(ids=ids)
            logger.info(f"✓ {len(ids)}개 문서 청크 삭제됨")
        except Exception as e:
            logger.error(f"✗ 문서 삭제 실패: {str(e)}")
            raise RuntimeError(f"ChromaDB 삭제 실패: {str(e)}")
    
    def delete_by_source(self, source: str) -> int:
        """
        특정 소스 파일의 모든 청크를 삭제합니다.
        
        Args:
            source: 소스 파일 경로
            
        Returns:
            삭제된 청크 개수
        """
        try:
            # 해당 소스의 모든 문서 조회
            results = self.collection.get(
                where={"source": source},
                include=[]
            )
            
            if results['ids'] is not None and len(results['ids']) > 0:
                self.delete(results['ids'])
                count = len(results['ids'])
                logger.info(f"✓ {source}의 {count}개 청크 삭제됨")
                return count
            else:
                logger.info(f"삭제할 문서 없음: {source}")
                return 0
        
        except Exception as e:
            logger.error(f"✗ 소스별 삭제 실패: {str(e)}")
            raise RuntimeError(f"ChromaDB 소스별 삭제 실패: {str(e)}")
    
    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        ChromaDB에서 ID로 문서를 조회합니다.
        
        Args:
            ids: 조회할 문서 청크 ID 목록
            
        Returns:
            문서 정보 목록
        """
        if not ids:
            return []
        
        try:
            results = self.collection.get(
                ids=ids,
                include=["documents", "metadatas", "embeddings"]
            )
            
            formatted_results = []
            if results['ids'] is not None and len(results['ids']) > 0:
                for i in range(len(results['ids'])):
                    formatted_results.append({
                        'id': results['ids'][i],
                        'document': results['documents'][i] if results['documents'] is not None else None,
                        'metadata': results['metadatas'][i] if results['metadatas'] is not None else {},
                        'embedding': results['embeddings'][i] if results['embeddings'] is not None else None
                    })
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"✗ 문서 조회 실패: {str(e)}")
            raise RuntimeError(f"ChromaDB 조회 실패: {str(e)}")
    
    def count(self) -> int:
        """
        ChromaDB의 총 문서 개수를 반환합니다.
        
        Returns:
            문서 개수
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"✗ 개수 조회 실패: {str(e)}")
            return 0
    
    def clear(self) -> None:
        """ChromaDB의 모든 데이터를 삭제합니다."""
        try:
            # 컬렉션 삭제 후 재생성
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"✓ 컬렉션 초기화됨: {self.collection_name}")
        except Exception as e:
            logger.error(f"✗ 컬렉션 초기화 실패: {str(e)}")
            raise RuntimeError(f"ChromaDB 초기화 실패: {str(e)}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        ChromaDB 컬렉션 정보를 반환합니다.
        
        Returns:
            컬렉션 통계 및 메타데이터
        """
        try:
            count = self.collection.count()
            
            # 고유 소스 파일 수 계산
            unique_sources = set()
            if count > 0:
                # 모든 메타데이터 조회 (페이징 처리)
                batch_size = 1000
                offset = 0
                
                while offset < count:
                    results = self.collection.get(
                        limit=batch_size,
                        offset=offset,
                        include=["metadatas"]
                    )
                    
                    if results['metadatas'] is not None and len(results['metadatas']) > 0:
                        for metadata in results['metadatas']:
                            if metadata and 'source' in metadata:
                                unique_sources.add(metadata['source'])
                    
                    offset += batch_size
            
            return {
                'collection_name': self.collection_name,
                'total_chunks': count,
                'total_documents': len(unique_sources),
                'persist_directory': str(self.persist_directory),
                'metadata': self.collection.metadata
            }
        
        except Exception as e:
            logger.error(f"✗ 컬렉션 정보 조회 실패: {str(e)}")
            return {
                'collection_name': self.collection_name,
                'error': str(e)
            }
    
    def get_all_sources(self) -> List[str]:
        """
        저장소의 모든 고유 소스 파일 목록을 반환합니다.
        
        Returns:
            소스 파일 경로 목록
        """
        try:
            count = self.collection.count()
            if count == 0:
                return []
            
            unique_sources = set()
            batch_size = 1000
            offset = 0
            
            while offset < count:
                results = self.collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["metadatas"]
                )
                
                if results['metadatas'] is not None and len(results['metadatas']) > 0:
                    for metadata in results['metadatas']:
                        if metadata and 'source' in metadata:
                            unique_sources.add(metadata['source'])
                
                offset += batch_size
            
            return sorted(list(unique_sources))
        
        except Exception as e:
            logger.error(f"✗ 소스 목록 조회 실패: {str(e)}")
            return []


# ============================================================================
# FAISS 구현 (향후 추가)
# ============================================================================

class FAISSVectorStore(VectorStore):
    """
    FAISS 기반 벡터 저장소 (향후 구현).
    
    평가 및 벤치마킹에 사용될 예정.
    - 더 빠른 검색 속도
    - 대규모 데이터셋 지원
    - GPU 가속 가능
    """
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FAISSVectorStore는 아직 구현되지 않았습니다. "
            "ChromaVectorStore를 사용하세요."
        )
    
    def add(self, ids, embeddings, documents, metadatas=None):
        raise NotImplementedError()
    
    def search(self, query_embedding, k=5, filter=None):
        raise NotImplementedError()
    
    def delete(self, ids):
        raise NotImplementedError()
    
    def get(self, ids):
        raise NotImplementedError()
    
    def count(self):
        raise NotImplementedError()
    
    def clear(self):
        raise NotImplementedError()
    
    def get_collection_info(self):
        raise NotImplementedError()


# ============================================================================
# 팩토리 함수
# ============================================================================

def get_vector_store(
    store_type: Optional[str] = None,
    **kwargs
) -> VectorStore:
    """
    설정에 따라 벡터 저장소 인스턴스를 생성합니다.
    
    Args:
        store_type: 저장소 타입 ("chroma" 또는 "faiss")
        **kwargs: 저장소별 추가 인자
        
    Returns:
        VectorStore 인스턴스
        
    Raises:
        ValueError: 지원하지 않는 저장소 타입
    """
    store_type = store_type or config.VECTOR_STORE.STORE_TYPE
    
    if store_type == "chroma":
        return ChromaVectorStore(**kwargs)
    elif store_type == "faiss":
        return FAISSVectorStore(**kwargs)
    else:
        raise ValueError(
            f"지원하지 않는 저장소 타입: {store_type}. "
            "'chroma' 또는 'faiss'를 사용하세요."
        )


# ============================================================================
# 전역 인스턴스 (싱글톤 패턴)
# ============================================================================

_vector_store_instance = None

def get_default_vector_store() -> VectorStore:
    """
    기본 벡터 저장소 싱글톤 인스턴스를 반환합니다.
    
    Returns:
        VectorStore 인스턴스
    """
    global _vector_store_instance
    
    if _vector_store_instance is None:
        _vector_store_instance = get_vector_store()
        logger.info("✓ 벡터 저장소 초기화됨")
    
    return _vector_store_instance
