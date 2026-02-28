"""
벡터 저장소 관리 모듈.
ChromaDB 및 FAISS 벡터 저장소를 위한 추상화 레이어.
"""

import logging
import json
import sqlite3
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path

import chromadb # type: ignore
from chromadb.config import Settings    # type: ignore
import numpy as np  # type: ignore

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

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
# FAISS 구현
# ============================================================================

class FAISSVectorStore(VectorStore):
    """
    FAISS 기반 벡터 저장소 (고성능 대규모 벡터 검색).
    
    특징:
    - 더 빠른 검색 속도 (100배~1000배)
    - 효율적인 메모리 사용
    - 대규모 데이터셋 지원 (수백만 개)
    
    메타데이터 저장:
    - SQLite: 빠른 조회를 위한 메타데이터 인덱싱
    - JSON: 백업 및 직렬화
    """
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[Path] = None,
        index_type: str = "ivf",
        use_gpu: bool = False
    ):
        """
        FAISS 벡터 저장소를 초기화합니다.
        
        Args:
            collection_name: 컬렉션 이름
            persist_directory: 영속성 디렉토리
            index_type: 인덱스 타입 ('flat', 'ivf', 'hnsw')
            use_gpu: GPU 사용 여부 (설치 필요)
        """
        if faiss is None:
            raise ImportError(
                "FAISS가 설치되지 않았습니다. "
                "다음 명령으로 설치하세요: pip install faiss-cpu"
            )
        
        self.collection_name = collection_name or config.VECTOR_STORE.CHROMA_COLLECTION_NAME
        self.persist_directory = persist_directory or config.VECTOR_STORE.PERSIST_DIR
        self.index_type = index_type
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        
        # 디렉토리 생성
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # 파일 경로
        self.index_path = self.persist_directory / f"{self.collection_name}.index"
        self.metadata_db = self.persist_directory / f"{self.collection_name}_metadata.db"
        self.id_mapping_file = self.persist_directory / f"{self.collection_name}_mapping.json"
        
        # FAISS 인덱스
        self.index = None
        self.embedding_dim = None
        
        # ID 매핑: FAISS 인덱스 위치 ↔ 문서 ID
        self.id_to_idx = {}  # 문서 ID → FAISS 인덱스 위치
        self.idx_to_id = {}  # FAISS 인덱스 위치 → 문서 ID
        self.next_idx = 0
        
        # SQLite 메타데이터 DB 초기화
        self._init_metadata_db()
        
        # 기존 인덱스 로드 또는 생성
        self._load_or_create_index()
        
        logger.info(f"✓ FAISS 벡터 저장소 초기화 완료: {self.collection_name}")
    
    def _init_metadata_db(self) -> None:
        """SQLite 메타데이터 데이터베이스를 초기화합니다."""
        try:
            conn = sqlite3.connect(str(self.metadata_db))
            cursor = conn.cursor()
            
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.collection_name}_metadata (
                    doc_id TEXT PRIMARY KEY,
                    document TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_doc_id 
                ON {self.collection_name}_metadata(doc_id)
            """)
            
            conn.commit()
            conn.close()
            logger.debug("✓ 메타데이터 DB 초기화 완료")
        except Exception as e:
            logger.error(f"✗ 메타데이터 DB 초기화 실패: {str(e)}")
            raise
    
    def _load_or_create_index(self) -> None:
        """기존 FAISS 인덱스를 로드하거나 새로 생성합니다."""
        try:
            # ID 매핑 로드
            if self.id_mapping_file.exists():
                with open(self.id_mapping_file, 'r') as f:
                    mapping = json.load(f)
                    self.id_to_idx = {k: int(v) for k, v in mapping.get('id_to_idx', {}).items()}
                    self.idx_to_id = {int(k): v for k, v in mapping.get('idx_to_id', {}).items()}
                    self.next_idx = mapping.get('next_idx', 0)
            
            # FAISS 인덱스 로드
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                self.embedding_dim = self.index.d
                logger.info(
                    f"✓ 기존 FAISS 인덱스 로드됨: "
                    f"{self.collection_name} (임베딩: {self.next_idx}개, "
                    f"차원: {self.embedding_dim})"
                )
            else:
                logger.info(f"새 FAISS 인덱스 생성 중... (첫 추가 시 차원 결정)")
        
        except Exception as e:
            logger.error(f"✗ 인덱스 로드 실패: {str(e)}")
            raise
    
    def _save_index(self) -> None:
        """FAISS 인덱스와 매핑을 디스크에 저장합니다."""
        try:
            if self.index is not None:
                faiss.write_index(self.index, str(self.index_path))
            
            # ID 매핑 저장
            with open(self.id_mapping_file, 'w') as f:
                json.dump({
                    'id_to_idx': self.id_to_idx,
                    'idx_to_id': self.idx_to_id,
                    'next_idx': self.next_idx
                }, f)
            
            logger.debug("✓ 인덱스 저장 완료")
        except Exception as e:
            logger.error(f"✗ 인덱스 저장 실패: {str(e)}")
            raise
    
    def _create_index_if_needed(self, embedding_dim: int) -> None:
        """첫 임베딩 추가 시 인덱스를 생성합니다."""
        if self.index is None:
            self.embedding_dim = embedding_dim
            
            if self.index_type == "flat":
                self.index = faiss.IndexFlatL2(embedding_dim)
            elif self.index_type == "ivf":
                # IVF: Inverted File Index (빠른 검색, 중간 메모리)
                # nlist: 클러스터 개수 (추천: sqrt(N))
                # 작은 데이터셋에서는 flat 사용
                quantizer = faiss.IndexFlatL2(embedding_dim)
                nlist = max(10, min(100, int(np.sqrt(embedding_dim))))
                self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
                self.index.nprobe = max(1, nlist // 10)
            elif self.index_type == "hnsw":
                # HNSW: 그래프 기반 (가장 빠름)
                self.index = faiss.IndexHNSWFlat(embedding_dim, 32)
            else:
                raise ValueError(f"지원하지 않는 인덱스 타입: {self.index_type}")
            
            logger.info(f"✓ FAISS 인덱스 생성: {self.index_type} (차원: {embedding_dim})")
    
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        벡터와 문서를 FAISS 저장소에 추가합니다.
        
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
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # 첫 추가 시 인덱스 생성
            if self.index is None:
                self._create_index_if_needed(embeddings_array.shape[1])
            
            # 기존 ID 처리 (업데이트)
            existing_ids = [doc_id for doc_id in ids if doc_id in self.id_to_idx]
            new_ids = [doc_id for doc_id in ids if doc_id not in self.id_to_idx]
            
            # 기존 ID 삭제 후 재추가
            if existing_ids:
                self.delete(existing_ids)
            
            # 새 벡터 추가
            if self.index_type == "ivf" and self.index.ntotal == 0:
                # IVF는 훈련 필요
                self.index.train(embeddings_array)
            
            # 벡터 추가
            for i, doc_id in enumerate(ids):
                self.index.add(np.array([embeddings_array[i]], dtype=np.float32))
                
                # ID 매핑 저장
                self.id_to_idx[doc_id] = self.next_idx
                self.idx_to_id[self.next_idx] = doc_id
                self.next_idx += 1
                
                # 메타데이터 저장
                self._save_metadata(
                    doc_id,
                    documents[i],
                    metadatas[i] if metadatas else None
                )
            
            self._save_index()
            logger.info(f"✓ {len(ids)}개 문서 청크 추가됨 (총: {self.next_idx}개)")
        
        except Exception as e:
            logger.error(f"✗ 문서 추가 실패: {str(e)}")
            raise RuntimeError(f"FAISS 추가 실패: {str(e)}")
    
    def _save_metadata(self, doc_id: str, document: str, metadata: Optional[Dict[str, Any]]) -> None:
        """메타데이터를 SQLite에 저장합니다."""
        try:
            conn = sqlite3.connect(str(self.metadata_db))
            cursor = conn.cursor()
            
            metadata_json = json.dumps(metadata or {})
            cursor.execute(f"""
                INSERT OR REPLACE INTO {self.collection_name}_metadata 
                (doc_id, document, metadata) 
                VALUES (?, ?, ?)
            """, (doc_id, document, metadata_json))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"✗ 메타데이터 저장 실패: {str(e)}")
            raise
    
    def _load_metadata(self, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """메타데이터를 SQLite에서 로드합니다."""
        try:
            conn = sqlite3.connect(str(self.metadata_db))
            cursor = conn.cursor()
            
            placeholders = ','.join('?' * len(doc_ids))
            cursor.execute(f"""
                SELECT doc_id, document, metadata FROM {self.collection_name}_metadata 
                WHERE doc_id IN ({placeholders})
            """, doc_ids)
            
            results = {}
            for row in cursor.fetchall():
                doc_id, document, metadata_json = row
                results[doc_id] = {
                    'document': document,
                    'metadata': json.loads(metadata_json) if metadata_json else {}
                }
            
            conn.close()
            return results
        except Exception as e:
            logger.error(f"✗ 메타데이터 로드 실패: {str(e)}")
            return {}
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        FAISS에서 유사도 검색을 수행합니다.
        
        Args:
            query_embedding: 쿼리 임베딩 벡터
            k: 반환할 결과 개수
            filter: 메타데이터 필터 (현재 포스트-필터링)
            
        Returns:
            검색 결과 목록
        """
        if not query_embedding or self.index is None:
            return []
        
        try:
            query_array = np.array([query_embedding], dtype=np.float32)
            k_actual = min(k * 2, self.next_idx) if filter else min(k, self.next_idx)
            
            # FAISS 검색
            distances, indices = self.index.search(query_array, k_actual)
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx == -1:  # Invalid index
                    continue
                
                doc_id = self.idx_to_id.get(int(idx))
                if not doc_id:
                    continue
                
                # 메타데이터 로드
                metadata_dict = self._load_metadata([doc_id])
                if doc_id not in metadata_dict:
                    continue
                
                doc_info = metadata_dict[doc_id]
                metadata = doc_info.get('metadata', {})
                
                # 필터 적용
                if filter:
                    if not self._matches_filter(metadata, filter):
                        continue
                
                # L2 거리를 유사도로 변환: similarity = 1 / (1 + distance)
                similarity = 1.0 / (1.0 + distance)
                
                results.append({
                    'id': doc_id,
                    'document': doc_info.get('document', ''),
                    'metadata': metadata,
                    'distance': float(distance),
                    'score': float(similarity)
                })
                
                if len(results) >= k:
                    break
            
            return results
        
        except Exception as e:
            logger.error(f"✗ 검색 실패: {str(e)}")
            raise RuntimeError(f"FAISS 검색 실패: {str(e)}")
    
    def _matches_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """메타데이터가 필터를 만족하는지 확인합니다."""
        for key, value in filter.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
    
    def delete(self, ids: List[str]) -> None:
        """
        FAISS에서 문서를 삭제합니다.
        
        Args:
            ids: 삭제할 문서 청크 ID 목록
        """
        if not ids or self.index is None:
            return
        
        try:
            # 삭제할 인덱스 정보 수집
            delete_indices = []
            for doc_id in ids:
                if doc_id in self.id_to_idx:
                    idx = self.id_to_idx[doc_id]
                    delete_indices.append(idx)
                    del self.id_to_idx[doc_id]
                    if idx in self.idx_to_id:
                        del self.idx_to_id[idx]
            
            if delete_indices:
                # SQLite에서 메타데이터 삭제
                conn = sqlite3.connect(str(self.metadata_db))
                cursor = conn.cursor()
                placeholders = ','.join('?' * len(ids))
                cursor.execute(f"""
                    DELETE FROM {self.collection_name}_metadata 
                    WHERE doc_id IN ({placeholders})
                """, ids)
                conn.commit()
                conn.close()
                
                # FAISS는 직접 삭제를 지원하지 않으므로,
                # ID 매핑을 업데이트하고 필요시 인덱스 재구성
                self._save_index()
                logger.info(f"✓ {len(ids)}개 문서 청크 삭제됨")
        
        except Exception as e:
            logger.error(f"✗ 문서 삭제 실패: {str(e)}")
            raise RuntimeError(f"FAISS 삭제 실패: {str(e)}")
    
    def delete_by_source(self, source: str) -> int:
        """
        특정 소스 파일의 모든 청크를 삭제합니다.
        
        Args:
            source: 소스 파일 경로
            
        Returns:
            삭제된 청크 개수
        """
        try:
            if self.metadata_db is None or not self.metadata_db.exists():
                return 0
            
            conn = sqlite3.connect(str(self.metadata_db))
            cursor = conn.cursor()
            
            cursor.execute(f"""
                SELECT doc_id FROM {self.collection_name}_metadata 
                WHERE metadata LIKE ?
            """, (f'%"source": "{source}"%',))
            
            doc_ids = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if doc_ids:
                self.delete(doc_ids)
                logger.info(f"✓ {source}의 {len(doc_ids)}개 청크 삭제됨")
                return len(doc_ids)
            
            return 0
        
        except Exception as e:
            logger.error(f"✗ 소스별 삭제 실패: {str(e)}")
            return 0
    
    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        ID로 문서를 조회합니다.
        
        Args:
            ids: 조회할 문서 청크 ID 목록
            
        Returns:
            문서 정보 목록
        """
        if not ids:
            return []
        
        try:
            metadata_dict = self._load_metadata(ids)
            
            results = []
            for doc_id in ids:
                if doc_id in metadata_dict:
                    doc_info = metadata_dict[doc_id]
                    results.append({
                        'id': doc_id,
                        'document': doc_info.get('document', ''),
                        'metadata': doc_info.get('metadata', {}),
                        'embedding': None  # FAISS는 저장 효율을 위해 임베딩 미반환
                    })
            
            return results
        
        except Exception as e:
            logger.error(f"✗ 문서 조회 실패: {str(e)}")
            raise RuntimeError(f"FAISS 조회 실패: {str(e)}")
    
    def count(self) -> int:
        """
        저장소의 총 문서 개수를 반환합니다.
        
        Returns:
            문서 개수
        """
        try:
            if self.metadata_db is None or not self.metadata_db.exists():
                return 0
            
            conn = sqlite3.connect(str(self.metadata_db))
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {self.collection_name}_metadata")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception:
            return 0
    
    def clear(self) -> None:
        """저장소의 모든 데이터를 삭제합니다."""
        try:
            # FAISS 인덱스 삭제
            if self.index_path.exists():
                self.index_path.unlink()
            
            # 메타데이터 DB 삭제
            if self.metadata_db.exists():
                self.metadata_db.unlink()
            
            # ID 매핑 파일 삭제
            if self.id_mapping_file.exists():
                self.id_mapping_file.unlink()
            
            # 메모리 상태 초기화
            self.index = None
            self.embedding_dim = None
            self.id_to_idx.clear()
            self.idx_to_id.clear()
            self.next_idx = 0
            
            # 메타데이터 DB 재생성
            self._init_metadata_db()
            
            logger.info(f"✓ 저장소 초기화됨: {self.collection_name}")
        
        except Exception as e:
            logger.error(f"✗ 저장소 초기화 실패: {str(e)}")
            raise RuntimeError(f"FAISS 초기화 실패: {str(e)}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        FAISS 컬렉션 정보를 반환합니다.
        
        Returns:
            컬렉션 통계 및 메타데이터
        """
        try:
            conn = sqlite3.connect(str(self.metadata_db))
            cursor = conn.cursor()
            
            # 전체 청크 수
            cursor.execute(f"SELECT COUNT(*) FROM {self.collection_name}_metadata")
            total_chunks = cursor.fetchone()[0]
            
            # 고유 소스 수
            cursor.execute(f"""
                SELECT COUNT(DISTINCT json_extract(metadata, '$.source')) 
                FROM {self.collection_name}_metadata
            """)
            total_documents = cursor.fetchone()[0] or 0
            
            conn.close()
            
            # 파일 크기
            index_size = self.index_path.stat().st_size if self.index_path.exists() else 0
            db_size = self.metadata_db.stat().st_size if self.metadata_db.exists() else 0
            
            return {
                'collection_name': self.collection_name,
                'total_chunks': total_chunks,
                'total_documents': total_documents,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type,
                'index_ntotal': self.index.ntotal if self.index else 0,
                'persist_directory': str(self.persist_directory),
                'index_file_size': f"{index_size / 1024 / 1024:.2f} MB",
                'metadata_db_size': f"{db_size / 1024 / 1024:.2f} MB",
                'total_size': f"{(index_size + db_size) / 1024 / 1024:.2f} MB"
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
            conn = sqlite3.connect(str(self.metadata_db))
            cursor = conn.cursor()
            
            cursor.execute(f"""
                SELECT DISTINCT json_extract(metadata, '$.source') 
                FROM {self.collection_name}_metadata
                WHERE json_extract(metadata, '$.source') IS NOT NULL
                ORDER BY json_extract(metadata, '$.source')
            """)
            
            sources = [row[0] for row in cursor.fetchall() if row[0]]
            conn.close()
            
            return sources
        
        except Exception as e:
            logger.error(f"✗ 소스 목록 조회 실패: {str(e)}")
            return []


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
        # FAISS 관련 설정 추가
        if 'index_type' not in kwargs:
            kwargs['index_type'] = config.VECTOR_STORE.FAISS_INDEX_TYPE
        if 'use_gpu' not in kwargs:
            kwargs['use_gpu'] = config.VECTOR_STORE.FAISS_USE_GPU
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
