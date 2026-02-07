"""
마크다운 문서의 임베딩 및 청킹 기능.
"""

import logging
import hashlib
from typing import List, Optional
from datetime import datetime
from pathlib import Path

from .config import config
from .ollama_client import get_ollama_client
from .schemas import DocumentChunk, DocumentMetadata

logger = logging.getLogger(__name__)


class TextChunker:
    """텍스트를 청크로 분할하는 클래스."""
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None
    ):
        """
        텍스트 청커를 초기화합니다.
        
        Args:
            chunk_size: 청크 크기 (문자 단위)
            chunk_overlap: 청크 간 겹침 크기
            separators: 분할 구분자 우선순위 목록
        """
        self.chunk_size = chunk_size or config.CHUNKING.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNKING.CHUNK_OVERLAP
        self.separators = separators or config.CHUNKING.SEPARATORS
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("청크 겹침은 청크 크기보다 작아야 합니다")
    
    def split_text(self, text: str) -> List[str]:
        """
        텍스트를 청크로 분할합니다.
        
        Args:
            text: 분할할 텍스트
            
        Returns:
            텍스트 청크 목록
        """
        if not text or not text.strip():
            return []
        
        # 재귀적으로 구분자를 사용하여 텍스트 분할
        chunks = self._split_text_recursive(text, self.separators)
        
        # 청크 크기 조정 및 병합
        final_chunks = self._merge_chunks(chunks)
        
        return final_chunks
    
    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """
        구분자 우선순위에 따라 재귀적으로 텍스트를 분할합니다.
        
        Args:
            text: 분할할 텍스트
            separators: 구분자 목록
            
        Returns:
            분할된 텍스트 조각 목록
        """
        if not separators:
            return [text]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator:
            splits = text.split(separator)
        else:
            # 빈 구분자는 문자 단위 분할
            return [text]
        
        # 각 분할된 부분이 아직 너무 크면 다음 구분자로 재귀
        result = []
        for split in splits:
            if len(split) > self.chunk_size and remaining_separators:
                result.extend(self._split_text_recursive(split, remaining_separators))
            elif split:
                result.append(split)
        
        return result
    
    def _merge_chunks(self, splits: List[str]) -> List[str]:
        """
        작은 조각들을 청크 크기에 맞게 병합합니다.
        
        Args:
            splits: 분할된 텍스트 조각
            
        Returns:
            병합된 청크 목록
        """
        if not splits:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = len(split)
            
            # 현재 청크에 추가 가능한지 확인
            if current_length + split_length <= self.chunk_size:
                current_chunk.append(split)
                current_length += split_length
            else:
                # 현재 청크 저장
                if current_chunk:
                    chunks.append("".join(current_chunk))
                
                # 새 청크 시작
                if split_length > self.chunk_size:
                    # 너무 큰 조각은 강제로 분할
                    chunks.extend(self._force_split(split))
                    current_chunk = []
                    current_length = 0
                else:
                    current_chunk = [split]
                    current_length = split_length
        
        # 마지막 청크 추가
        if current_chunk:
            chunks.append("".join(current_chunk))
        
        return chunks
    
    def _force_split(self, text: str) -> List[str]:
        """
        너무 큰 텍스트를 강제로 분할합니다.
        
        Args:
            text: 분할할 텍스트
            
        Returns:
            분할된 청크 목록
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        
        return chunks
    
    def split_text_with_overlap(self, text: str) -> List[str]:
        """
        오버랩을 적용하여 텍스트를 분할합니다.
        
        Args:
            text: 분할할 텍스트
            
        Returns:
            오버랩이 적용된 청크 목록
        """
        base_chunks = self.split_text(text)
        
        if not base_chunks or self.chunk_overlap == 0:
            return base_chunks
        
        # 오버랩 적용
        overlapped_chunks = []
        for i, chunk in enumerate(base_chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # 이전 청크의 마지막 부분과 현재 청크 결합
                prev_chunk = base_chunks[i - 1]
                overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
                overlapped_chunks.append(overlap_text + chunk)
        
        return overlapped_chunks


class MarkdownChunker(TextChunker):
    """마크다운 구조를 인식하는 텍스트 청커."""
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        """
        마크다운 청커를 초기화합니다.
        
        Args:
            chunk_size: 청크 크기
            chunk_overlap: 청크 간 겹침
        """
        # 마크다운 구조를 고려한 구분자
        md_separators = [
            "\n## ",      # 2차 헤더
            "\n### ",     # 3차 헤더
            "\n#### ",    # 4차 헤더
            "\n\n",       # 문단
            "\n",         # 줄
            ". ",         # 문장
            " ",          # 단어
            ""            # 문자
        ]
        
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=md_separators
        )
    
    def split_text(self, text: str) -> List[str]:
        """
        마크다운 텍스트를 구조를 고려하여 분할합니다.
        
        Args:
            text: 마크다운 텍스트
            
        Returns:
            청크 목록
        """
        # 1차 헤더(#)로 먼저 분할
        sections = text.split("\n# ")
        
        chunks = []
        for i, section in enumerate(sections):
            # 첫 번째가 아니면 헤더 복원
            if i > 0:
                section = "# " + section
            
            # 각 섹션을 부모 클래스의 분할 로직으로 처리
            section_chunks = super().split_text(section)
            chunks.extend(section_chunks)
        
        return chunks


class DocumentEmbedder:
    """문서를 임베딩으로 변환하는 클래스."""
    
    def __init__(
        self,
        chunker: Optional[TextChunker] = None,
        use_markdown_chunker: bool = True
    ):
        """
        문서 임베더를 초기화합니다.
        
        Args:
            chunker: 사용할 청커 (None이면 기본 청커 생성)
            use_markdown_chunker: 마크다운 전용 청커 사용 여부
        """
        if chunker:
            self.chunker = chunker
        elif use_markdown_chunker:
            self.chunker = MarkdownChunker()
        else:
            self.chunker = TextChunker()
        
        self.ollama_client = get_ollama_client()
    
    def embed_document(
        self,
        content: str,
        source: str,
        show_progress: bool = False
    ) -> List[DocumentChunk]:
        """
        문서를 청크로 분할하고 임베딩을 생성합니다.
        
        Args:
            content: 문서 내용
            source: 문서 원본 경로
            show_progress: 진행 상황 로깅 여부
            
        Returns:
            임베딩이 포함된 문서 청크 목록
        """
        # 텍스트 청킹
        chunks_text = self.chunker.split_text(content)
        
        if not chunks_text:
            logger.warning(f"문서에서 청크를 생성하지 못했습니다: {source}")
            return []
        
        if show_progress:
            logger.info(f"문서를 {len(chunks_text)}개 청크로 분할: {source}")
        
        # 현재 시간
        created_at = datetime.utcnow().isoformat()
        
        # 각 청크에 대해 임베딩 생성
        document_chunks = []
        total_chunks = len(chunks_text)
        
        for idx, chunk_text in enumerate(chunks_text):
            if show_progress and (idx % 10 == 0 or idx == total_chunks - 1):
                logger.info(f"임베딩 생성 중: {idx + 1}/{total_chunks}")
            
            try:
                # 임베딩 생성
                embedding = self.ollama_client.embed(chunk_text)
                
                # 청크 ID 생성 (source와 chunk_id 기반 해시)
                chunk_id_str = f"{source}_{idx}"
                chunk_hash = hashlib.md5(chunk_id_str.encode()).hexdigest()
                
                # 메타데이터 생성
                metadata = DocumentMetadata(
                    source=source,
                    chunk_id=idx,
                    total_chunks=total_chunks,
                    created_at=created_at
                )
                
                # DocumentChunk 생성
                doc_chunk = DocumentChunk(
                    id=chunk_hash,
                    content=chunk_text,
                    metadata=metadata,
                    embedding=embedding
                )
                
                document_chunks.append(doc_chunk)
                
            except Exception as e:
                logger.error(f"청크 {idx} 임베딩 실패: {str(e)}")
                continue
        
        if show_progress:
            logger.info(f"임베딩 생성 완료: {len(document_chunks)}/{total_chunks} 성공")
        
        return document_chunks
    
    def embed_document_from_file(
        self,
        file_path: Path,
        show_progress: bool = False
    ) -> List[DocumentChunk]:
        """
        파일에서 문서를 읽어 임베딩을 생성합니다.
        
        Args:
            file_path: 문서 파일 경로
            show_progress: 진행 상황 로깅 여부
            
        Returns:
            임베딩이 포함된 문서 청크 목록
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            source = str(file_path)
            return self.embed_document(content, source, show_progress)
            
        except Exception as e:
            logger.error(f"파일 읽기 실패 {file_path}: {str(e)}")
            return []
    
    def embed_batch(
        self,
        documents: List[tuple[str, str]],
        show_progress: bool = True
    ) -> dict[str, List[DocumentChunk]]:
        """
        여러 문서를 배치로 임베딩합니다.
        
        Args:
            documents: (content, source) 튜플 목록
            show_progress: 진행 상황 로깅 여부
            
        Returns:
            source를 키로 하는 DocumentChunk 목록 딕셔너리
        """
        results = {}
        total_docs = len(documents)
        
        for idx, (content, source) in enumerate(documents):
            if show_progress:
                logger.info(f"문서 처리 중: {idx + 1}/{total_docs} - {source}")
            
            chunks = self.embed_document(content, source, show_progress=False)
            results[source] = chunks
        
        if show_progress:
            total_chunks = sum(len(chunks) for chunks in results.values())
            logger.info(f"배치 임베딩 완료: {len(results)} 문서, {total_chunks} 청크")
        
        return results


def create_embedder(use_markdown: bool = True) -> DocumentEmbedder:
    """
    문서 임베더를 생성합니다.
    
    Args:
        use_markdown: 마크다운 전용 청커 사용 여부
        
    Returns:
        DocumentEmbedder 인스턴스
    """
    return DocumentEmbedder(use_markdown_chunker=use_markdown)


def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    use_markdown: bool = True
) -> List[str]:
    """
    텍스트를 청크로 분할하는 헬퍼 함수.
    
    Args:
        text: 분할할 텍스트
        chunk_size: 청크 크기
        chunk_overlap: 청크 간 겹침
        use_markdown: 마크다운 청커 사용 여부
        
    Returns:
        텍스트 청크 목록
    """
    if use_markdown:
        chunker = MarkdownChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    return chunker.split_text(text)
