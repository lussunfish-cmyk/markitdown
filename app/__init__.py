"""
MarkItDown RAG 패키지.

이 패키지는 문서 변환, 벡터 인덱싱, RAG 기반 질의응답 기능을 제공합니다.
"""

from .config import config, validate_config
from .ollama_client import OllamaClient, get_ollama_client
from .vector_store import VectorStore, ChromaVectorStore, get_vector_store
from .embedding import TextChunker, MarkdownChunker, DocumentEmbedder, create_embedder
from .retriever import (
    SearchResult,
    BaseRetriever,
    VectorRetriever,
    BM25Retriever,
    HybridRetriever,
    AdvancedRetriever,
    get_retriever,
    create_retriever
)
from .rag import RAGPipeline, RAGResult, RAGMetrics, get_rag_pipeline
from .indexer import DocumentIndexer, IndexStateManager
from .batch_manager import BatchStateManager
from .schemas import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    RAGRequest,
    RAGResponse,
    IndexRequest,
    IndexResponse,
    BatchJobResponse
)

__version__ = config.API.VERSION

__all__ = [
    # 설정
    "config",
    "validate_config",
    
    # Ollama 클라이언트
    "OllamaClient",
    "get_ollama_client",
    
    # 벡터 저장소
    "VectorStore",
    "ChromaVectorStore",
    "get_vector_store",
    
    # 임베딩 및 청킹
    "TextChunker",
    "MarkdownChunker",
    "DocumentEmbedder",
    "create_embedder",
    
    # 검색
    "SearchResult",
    "BaseRetriever",
    "VectorRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "AdvancedRetriever",
    "get_retriever",
    "create_retriever",
    
    # RAG
    "RAGPipeline",
    "RAGResult",
    "RAGMetrics",
    "get_rag_pipeline",
    
    # 인덱싱
    "DocumentIndexer",
    "IndexStateManager",
    
    # 배치 관리
    "BatchStateManager",
    
    # 스키마
    "Document",
    "DocumentChunk",
    "DocumentMetadata",
    "RAGRequest",
    "RAGResponse",
    "IndexRequest",
    "IndexResponse",
    "BatchJobResponse",
]
