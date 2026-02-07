"""
문서 자동 인덱싱 모듈.
output 폴더의 마크다운 파일을 자동으로 벡터 스토어에 인덱싱합니다.
"""

import logging
import hashlib
import json
import time
from typing import List, Dict, Optional, Set, Any
from pathlib import Path
from datetime import datetime

from .config import config
from .embedding import DocumentEmbedder, create_embedder
from .vector_store import get_vector_store, VectorStore
from .schemas import Document, IndexResponse

logger = logging.getLogger(__name__)


# ============================================================================
# 인덱스 상태 관리
# ============================================================================

class IndexStateManager:
    """인덱싱 상태를 추적하고 관리하는 클래스."""
    
    def __init__(self, state_file: Optional[Path] = None):
        """
        상태 관리자를 초기화합니다.
        
        Args:
            state_file: 상태 파일 경로 (None이면 기본값 사용)
        """
        self.state_file = state_file or config.INDEXING.INDEX_STATE_FILE
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """
        디스크에서 인덱싱 상태를 로드합니다.
        
        Returns:
            상태 딕셔너리
        """
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"상태 파일 로드 실패: {e}")
                return {"files": {}, "metadata": {}}
        return {"files": {}, "metadata": {}}
    
    def save_state(self) -> None:
        """현재 상태를 디스크에 저장합니다."""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
            logger.debug(f"상태 저장됨: {self.state_file}")
        except Exception as e:
            logger.error(f"상태 저장 실패: {e}")
    
    def is_indexed(self, file_path: Path) -> bool:
        """
        파일이 이미 인덱싱되었는지 확인합니다.
        
        Args:
            file_path: 파일 경로
            
        Returns:
            인덱싱 여부
        """
        file_key = str(file_path.absolute())
        return file_key in self.state["files"]
    
    def needs_reindex(self, file_path: Path) -> bool:
        """
        파일이 재인덱싱이 필요한지 확인합니다 (수정된 경우).
        
        Args:
            file_path: 파일 경로
            
        Returns:
            재인덱싱 필요 여부
        """
        if not self.is_indexed(file_path):
            return True
        
        file_key = str(file_path.absolute())
        stored_hash = self.state["files"][file_key].get("hash")
        current_hash = self._compute_file_hash(file_path)
        
        return stored_hash != current_hash
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """
        파일의 SHA256 해시를 계산합니다.
        
        Args:
            file_path: 파일 경로
            
        Returns:
            해시 문자열
        """
        try:
            content = file_path.read_bytes()
            return hashlib.sha256(content).hexdigest()
        except Exception as e:
            logger.error(f"파일 해시 계산 실패 ({file_path}): {e}")
            return ""
    
    def add_file(
        self,
        file_path: Path,
        chunk_ids: List[str],
        num_chunks: int,
        status: str = "indexed"
    ) -> None:
        """
        파일을 인덱싱 상태에 추가합니다.
        
        Args:
            file_path: 파일 경로
            chunk_ids: 청크 ID 목록
            num_chunks: 청크 개수
            status: 인덱싱 상태
        """
        file_key = str(file_path.absolute())
        self.state["files"][file_key] = {
            "path": str(file_path),
            "filename": file_path.name,
            "hash": self._compute_file_hash(file_path),
            "chunk_ids": chunk_ids,
            "num_chunks": num_chunks,
            "status": status,
            "indexed_at": datetime.now().isoformat()
        }
        self.save_state()
    
    def remove_file(self, file_path: Path) -> Optional[List[str]]:
        """
        인덱싱 상태에서 파일을 제거합니다.
        
        Args:
            file_path: 파일 경로
            
        Returns:
            제거된 청크 ID 목록 (없으면 None)
        """
        file_key = str(file_path.absolute())
        if file_key in self.state["files"]:
            chunk_ids = self.state["files"][file_key].get("chunk_ids", [])
            del self.state["files"][file_key]
            self.save_state()
            return chunk_ids
        return None
    
    def get_indexed_files(self) -> List[Dict[str, Any]]:
        """
        인덱싱된 모든 파일 목록을 반환합니다.
        
        Returns:
            파일 정보 딕셔너리 목록
        """
        return list(self.state["files"].values())
    
    def get_file_info(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        특정 파일의 인덱싱 정보를 반환합니다.
        
        Args:
            file_path: 파일 경로
            
        Returns:
            파일 정보 딕셔너리 (없으면 None)
        """
        file_key = str(file_path.absolute())
        return self.state["files"].get(file_key)
    
    def clear(self) -> None:
        """모든 인덱싱 상태를 초기화합니다."""
        self.state = {"files": {}, "metadata": {}}
        self.save_state()


# ============================================================================
# 문서 인덱서
# ============================================================================

class DocumentIndexer:
    """문서를 벡터 스토어에 인덱싱하는 클래스."""
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[DocumentEmbedder] = None,
        state_manager: Optional[IndexStateManager] = None,
        document_dir: Optional[Path] = None
    ):
        """
        문서 인덱서를 초기화합니다.
        
        Args:
            vector_store: 벡터 저장소 (None이면 기본값 사용)
            embedder: 문서 임베더 (None이면 기본값 사용)
            state_manager: 상태 관리자 (None이면 새로 생성)
            document_dir: 문서 디렉토리 (None이면 기본값 사용)
        """
        self.vector_store = vector_store or get_vector_store()
        self.embedder = embedder or create_embedder()
        self.state_manager = state_manager or IndexStateManager()
        self.document_dir = document_dir or config.INDEXING.DOCUMENT_DIR
        self.supported_formats = config.INDEXING.SUPPORTED_FORMATS
        
        # 문서 디렉토리 생성
        self.document_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✓ DocumentIndexer 초기화됨")
        logger.info(f"  문서 디렉토리: {self.document_dir}")
        logger.info(f"  지원 형식: {self.supported_formats}")
    
    def index_file(
        self,
        file_path: Path,
        force_reindex: bool = False
    ) -> Dict[str, Any]:
        """
        단일 파일을 인덱싱합니다.
        
        Args:
            file_path: 파일 경로
            force_reindex: 이미 인덱싱된 파일도 강제로 재인덱싱
            
        Returns:
            인덱싱 결과 딕셔너리
        """
        result = {
            "filename": file_path.name,
            "path": str(file_path),
            "status": "skipped",
            "chunks": 0,
            "message": ""
        }
        
        # 파일 존재 확인
        if not file_path.exists():
            result["status"] = "failed"
            result["message"] = "파일이 존재하지 않음"
            logger.warning(f"파일 없음: {file_path}")
            return result
        
        # 형식 지원 확인
        if file_path.suffix.lower() not in self.supported_formats:
            result["status"] = "failed"
            result["message"] = f"지원하지 않는 형식: {file_path.suffix}"
            logger.warning(f"지원하지 않는 형식: {file_path}")
            return result
        
        # 재인덱싱 필요 여부 확인
        if not force_reindex and not self.state_manager.needs_reindex(file_path):
            result["status"] = "skipped"
            result["message"] = "이미 인덱싱됨 (변경 없음)"
            file_info = self.state_manager.get_file_info(file_path)
            result["chunks"] = file_info.get("num_chunks", 0) if file_info else 0
            logger.debug(f"스킵: {file_path.name}")
            return result
        
        # 기존 청크 삭제 (재인덱싱인 경우)
        if self.state_manager.is_indexed(file_path):
            old_chunk_ids = self.state_manager.remove_file(file_path)
            if old_chunk_ids:
                try:
                    self.vector_store.delete(old_chunk_ids)
                    logger.debug(f"기존 청크 삭제: {len(old_chunk_ids)}개")
                except Exception as e:
                    logger.warning(f"기존 청크 삭제 실패: {e}")
        
        # 파일 읽기 및 인덱싱
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # 문서 임베딩
            chunks = self.embedder.embed_document(
                content=content,
                source=str(file_path.relative_to(self.document_dir))
            )
            
            if not chunks:
                result["status"] = "failed"
                result["message"] = "청크 생성 실패 (빈 문서)"
                return result
            
            # 벡터 스토어에 추가
            ids = [chunk.id for chunk in chunks]
            embeddings = [chunk.embedding for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata.dict() for chunk in chunks]
            
            self.vector_store.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            # 상태 저장
            chunk_ids = [chunk.id for chunk in chunks]
            self.state_manager.add_file(
                file_path=file_path,
                chunk_ids=chunk_ids,
                num_chunks=len(chunks),
                status="indexed"
            )
            
            result["status"] = "indexed"
            result["chunks"] = len(chunks)
            result["message"] = f"성공적으로 인덱싱됨 ({len(chunks)} 청크)"
            logger.info(f"✓ 인덱싱 완료: {file_path.name} ({len(chunks)} 청크)")
            
        except Exception as e:
            result["status"] = "failed"
            result["message"] = f"인덱싱 에러: {str(e)}"
            logger.error(f"✗ 인덱싱 실패 ({file_path.name}): {e}")
        
        return result
    
    def index_directory(
        self,
        directory: Optional[Path] = None,
        force_reindex: bool = False,
        recursive: bool = False
    ) -> IndexResponse:
        """
        디렉토리의 모든 지원 파일을 인덱싱합니다.
        
        Args:
            directory: 디렉토리 경로 (None이면 기본 문서 디렉토리)
            force_reindex: 이미 인덱싱된 파일도 강제로 재인덱싱
            recursive: 하위 디렉토리도 포함
            
        Returns:
            인덱싱 결과
        """
        directory = directory or self.document_dir
        
        logger.info("\n" + "="*70)
        logger.info(f"디렉토리 인덱싱 시작: {directory}")
        logger.info("="*70)
        
        # 파일 수집
        files = self._collect_files(directory, recursive)
        
        if not files:
            logger.warning("인덱싱할 파일이 없음")
            return IndexResponse(
                total_files=0,
                indexed_files=0,
                failed_files=0,
                total_chunks=0,
                files=[]
            )
        
        logger.info(f"발견된 파일: {len(files)}개")
        
        # 각 파일 인덱싱
        results = []
        indexed_count = 0
        failed_count = 0
        total_chunks = 0
        
        for i, file_path in enumerate(files, 1):
            logger.info(f"[{i}/{len(files)}] 처리 중: {file_path.name}")
            result = self.index_file(file_path, force_reindex)
            results.append(result)
            
            if result["status"] == "indexed":
                indexed_count += 1
                total_chunks += result["chunks"]
            elif result["status"] == "failed":
                failed_count += 1
            elif result["status"] == "skipped":
                # 스킵된 파일도 청크 수 집계
                total_chunks += result["chunks"]
        
        logger.info("\n" + "="*70)
        logger.info("인덱싱 완료")
        logger.info(f"  총 파일: {len(files)}")
        logger.info(f"  인덱싱됨: {indexed_count}")
        logger.info(f"  실패: {failed_count}")
        logger.info(f"  스킵됨: {len(files) - indexed_count - failed_count}")
        logger.info(f"  총 청크: {total_chunks}")
        logger.info("="*70)
        
        return IndexResponse(
            total_files=len(files),
            indexed_files=indexed_count,
            failed_files=failed_count,
            total_chunks=total_chunks,
            files=results
        )
    
    def _collect_files(self, directory: Path, recursive: bool) -> List[Path]:
        """
        디렉토리에서 지원하는 파일을 수집합니다.
        
        Args:
            directory: 디렉토리 경로
            recursive: 재귀 검색 여부
            
        Returns:
            파일 경로 리스트
        """
        files = []
        
        if recursive:
            for ext in self.supported_formats:
                files.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in self.supported_formats:
                files.extend(directory.glob(f"*{ext}"))
        
        return sorted(files)
    
    def delete_document(self, file_path: Path) -> bool:
        """
        문서를 벡터 스토어에서 삭제합니다.
        
        Args:
            file_path: 파일 경로
            
        Returns:
            삭제 성공 여부
        """
        try:
            chunk_ids = self.state_manager.remove_file(file_path)
            
            if chunk_ids:
                self.vector_store.delete(chunk_ids)
                logger.info(f"✓ 문서 삭제됨: {file_path.name} ({len(chunk_ids)} 청크)")
                return True
            else:
                logger.warning(f"삭제할 문서 없음: {file_path.name}")
                return False
                
        except Exception as e:
            logger.error(f"✗ 문서 삭제 실패 ({file_path.name}): {e}")
            return False
    
    def delete_by_source(self, source: str) -> bool:
        """
        소스명으로 문서를 삭제합니다.
        
        Args:
            source: 소스 파일명 또는 경로
            
        Returns:
            삭제 성공 여부
        """
        try:
            # 상태에서 파일 찾기
            indexed_files = self.state_manager.get_indexed_files()
            
            for file_info in indexed_files:
                if file_info["filename"] == source or source in file_info["path"]:
                    file_path = Path(file_info["path"])
                    return self.delete_document(file_path)
            
            logger.warning(f"삭제할 문서 없음: {source}")
            return False
            
        except Exception as e:
            logger.error(f"✗ 문서 삭제 실패 ({source}): {e}")
            return False
    
    def rebuild_index(self) -> IndexResponse:
        """
        전체 인덱스를 재구성합니다 (모든 파일 재인덱싱).
        
        Returns:
            인덱싱 결과
        """
        logger.info("\n" + "="*70)
        logger.info("전체 인덱스 재구성 시작")
        logger.info("="*70)
        
        # 기존 인덱스 삭제
        try:
            # 벡터 스토어 초기화 (모든 데이터 삭제)
            logger.info("기존 인덱스 삭제 중...")
            # ChromaDB의 경우 collection을 삭제하고 재생성
            if hasattr(self.vector_store, 'collection'):
                # 새 벡터 스토어 생성 (기존 것 초기화)
                self.vector_store = get_vector_store()
            
            # 상태 초기화
            self.state_manager.clear()
            logger.info("✓ 기존 인덱스 삭제 완료")
            
        except Exception as e:
            logger.error(f"기존 인덱스 삭제 실패: {e}")
        
        # 전체 재인덱싱
        return self.index_directory(force_reindex=True)
    
    def get_indexed_files(self) -> List[Document]:
        """
        인덱싱된 모든 파일 목록을 반환합니다.
        
        Returns:
            Document 객체 리스트
        """
        files = self.state_manager.get_indexed_files()
        
        documents = []
        for file_info in files:
            doc = Document(
                id=Path(file_info["path"]).stem,
                filename=file_info["filename"],
                total_chunks=file_info["num_chunks"],
                indexed_at=file_info.get("indexed_at", ""),
                status=file_info.get("status", "unknown")
            )
            documents.append(doc)
        
        return documents
    
    def get_stats(self) -> Dict[str, Any]:
        """
        인덱싱 통계를 반환합니다.
        
        Returns:
            통계 딕셔너리
        """
        files = self.state_manager.get_indexed_files()
        
        total_files = len(files)
        total_chunks = sum(f.get("num_chunks", 0) for f in files)
        
        status_counts = {}
        for f in files:
            status = f.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_files": total_files,
            "total_chunks": total_chunks,
            "status": status_counts,
            "vector_store_count": self.vector_store.count(),
            "files": [f["filename"] for f in files]
        }


# ============================================================================
# 팩토리 함수
# ============================================================================

_indexer_instance: Optional[DocumentIndexer] = None


def get_indexer(
    vector_store: Optional[VectorStore] = None,
    embedder: Optional[DocumentEmbedder] = None
) -> DocumentIndexer:
    """
    DocumentIndexer 싱글톤 인스턴스를 반환합니다.
    
    Args:
        vector_store: 벡터 저장소 (None이면 기본값 사용)
        embedder: 문서 임베더 (None이면 기본값 사용)
        
    Returns:
        DocumentIndexer 인스턴스
    """
    global _indexer_instance
    
    if _indexer_instance is None:
        _indexer_instance = DocumentIndexer(
            vector_store=vector_store,
            embedder=embedder
        )
    
    return _indexer_instance


def create_indexer(
    vector_store: Optional[VectorStore] = None,
    embedder: Optional[DocumentEmbedder] = None,
    document_dir: Optional[Path] = None
) -> DocumentIndexer:
    """
    새로운 DocumentIndexer 인스턴스를 생성합니다.
    
    Args:
        vector_store: 벡터 저장소 (None이면 기본값 사용)
        embedder: 문서 임베더 (None이면 기본값 사용)
        document_dir: 문서 디렉토리 (None이면 기본값 사용)
        
    Returns:
        새로운 DocumentIndexer 인스턴스
    """
    return DocumentIndexer(
        vector_store=vector_store,
        embedder=embedder,
        document_dir=document_dir
    )
