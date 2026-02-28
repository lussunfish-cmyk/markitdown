"""
마크다운 문서의 임베딩 및 청킹 기능.
"""

import logging
import hashlib
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from .config import config
from .llm_client import get_llm_client
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
    """마크다운 구조를 인식하는 향상된 텍스트 청커.
    
    개선 사항:
    - 헤더 기반 섹션 분할
    - 각 청크에 상위 헤더 컨텍스트 추가
    - 문장 경계 존중
    - 의미론적으로 일관된 청크 생성
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        add_header_context: bool = True
    ):
        """
        마크다운 청커를 초기화합니다.
        
        Args:
            chunk_size: 청크 크기
            chunk_overlap: 청크 간 겹침
            add_header_context: 각 청크에 상위 헤더 추가 여부
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=config.CHUNKING.MD_SEPARATORS
        )
        self.add_header_context = add_header_context
    
    def split_text(self, text: str) -> List[str]:
        """
        마크다운 텍스트를 구조를 고려하여 분할합니다.
        헤더 계층을 추출하여 각 청크에 컨텍스트를 추가합니다.
        
        Args:
            text: 마크다운 텍스트
            
        Returns:
            청크 목록 (헤더 컨텍스트 포함)
        """
        if not self.add_header_context:
            # 기본 분할만 수행
            return self._basic_split(text)
        
        # 헤더 기반 섹션으로 분할
        sections = self._extract_sections(text)
        
        # 각 섹션을 청크로 변환
        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(section)
            chunks.extend(section_chunks)
        
        return chunks
    
    def _basic_split(self, text: str) -> List[str]:
        """기본 분할 (헤더 컨텍스트 없음)"""
        sections = text.split("\n# ")
        chunks = []
        for i, section in enumerate(sections):
            if i > 0:
                section = "# " + section
            section_chunks = super().split_text(section)
            chunks.extend(section_chunks)
        return chunks
    
    def _extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        텍스트를 헤더 기반 섹션으로 분할하고 계층 정보를 추출합니다.
        
        Returns:
            [{
                'level': 헤더 레벨,
                'title': 헤더 제목,
                'content': 섹션 내용,
                'parent_headers': 상위 헤더 리스트
            }]
        """
        import re
        
        lines = text.split('\n')
        sections = []
        current_headers = {}  # level -> title 매핑
        current_section = {'level': 0, 'title': '', 'content': [], 'parent_headers': []}
        
        for line in lines:
            # 헤더 감지
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if header_match:
                # 이전 섹션 저장
                if current_section['content']:
                    current_section['content'] = '\n'.join(current_section['content'])
                    sections.append(current_section.copy())
                
                # 새 섹션 시작
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # 현재 레벨 이상의 헤더 제거
                current_headers = {k: v for k, v in current_headers.items() if k < level}
                current_headers[level] = title
                
                # 상위 헤더 목록 생성
                parent_headers = [current_headers[l] for l in sorted(current_headers.keys()) if l < level]
                
                current_section = {
                    'level': level,
                    'title': title,
                    'content': [line],
                    'parent_headers': parent_headers
                }
            else:
                # 일반 텍스트
                current_section['content'].append(line)
        
        # 마지막 섹션 저장
        if current_section['content']:
            current_section['content'] = '\n'.join(current_section['content'])
            sections.append(current_section)
        
        return sections
    
    def _chunk_section(self, section: Dict[str, Any]) -> List[str]:
        """
        섹션을 청크로 분할하고 각 청크에 컨텍스트를 추가합니다.
        
        Args:
            section: 섹션 정보
            
        Returns:
            컨텍스트가 포함된 청크 목록
        """
        content = section['content']
        parent_headers = section['parent_headers']
        title = section['title']
        
        # 컨텍스트 헤더 생성 (상위 헤더들)
        context_header = ""
        if parent_headers:
            context_header = " > ".join(parent_headers)
            if title:
                context_header += " > " + title
        elif title:
            context_header = title
        
        # 섹션이 충분히 작으면 그대로 반환
        if len(content) <= self.chunk_size:
            if context_header:
                return [f"[{context_header}]\n\n{content}"]
            return [content]
        
        # 큰 섹션은 분할
        base_chunks = super().split_text(content)
        
        # 각 청크에 컨텍스트 추가
        if context_header:
            context_prefix = f"[{context_header}]\n\n"
            # 컨텍스트 길이를 고려하여 청크 크기 조정
            chunks_with_context = []
            for chunk in base_chunks:
                # 컨텍스트가 너무 길어지지 않도록 청크 일부를 줄일 수 있음
                available_size = self.chunk_size - len(context_prefix)
                if available_size > 200:  # 최소 200자는 보장
                    chunk = chunk[:available_size]
                chunks_with_context.append(context_prefix + chunk)
            return chunks_with_context
        
        return base_chunks


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
        
        self.llm_client = get_llm_client()
    
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
                embedding = self.llm_client.embed(chunk_text)
                
                # 청크 ID 생성 (source와 chunk_id 기반 해시)
                chunk_id_str = f"{source}_{idx}"
                chunk_hash = hashlib.md5(chunk_id_str.encode()).hexdigest()
                
                # 청크에서 컨텍스트 헤더 추출 (있는 경우)
                section_title = self._extract_section_title(chunk_text)
                
                # 메타데이터 생성
                metadata = DocumentMetadata(
                    source=source,
                    chunk_id=idx,
                    total_chunks=total_chunks,
                    created_at=created_at
                )
                
                # 섹션 제목이 있으면 메타데이터에 추가
                if section_title:
                    metadata.section_title = section_title
                
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
    
    def _extract_section_title(self, chunk_text: str) -> Optional[str]:
        """
        청크에서 섹션 제목을 추출합니다.
        
        Args:
            chunk_text: 청크 텍스트
            
        Returns:
            섹션 제목 (없으면 None)
        """
        import re
        
        # [제목] 형식의 컨텍스트 헤더 추출
        context_match = re.match(r'^\[(.+?)\]', chunk_text)
        if context_match:
            return context_match.group(1)
        
        # 일반 마크다운 헤더 추출 (첫 번째 헤더)
        header_match = re.search(r'^#{1,6}\s+(.+)$', chunk_text, re.MULTILINE)
        if header_match:
            return header_match.group(1).strip()
        
        return None
    
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
