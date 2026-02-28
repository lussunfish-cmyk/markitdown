"""
LLM 클라이언트 추상 인터페이스.
다양한 LLM 백엔드(Ollama, LM Studio 등)를 지원하기 위한 추상화 레이어.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Generator

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """LLM 서버와 상호작용하는 클라이언트 추상 인터페이스."""
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        텍스트를 위한 임베딩을 생성합니다.
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            실수 목록으로 된 임베딩 벡터
            
        Raises:
            RuntimeError: 임베딩 생성이 실패한 경우
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        """
        여러 텍스트에 대한 임베딩을 생성합니다.
        
        Args:
            texts: 임베딩할 텍스트 목록
            show_progress: 진행 상황 기록 여부
            
        Returns:
            임베딩 벡터 목록
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        num_predict: Optional[int] = None
    ) -> str:
        """
        LLM을 사용하여 텍스트를 생성합니다.
        
        Args:
            prompt: 입력 프롬프트
            temperature: 샘플링 온도
            top_p: Top-p 샘플링 파라미터
            num_predict: 생성할 최대 토큰 수
            
        Returns:
            생성된 텍스트
            
        Raises:
            RuntimeError: 생성이 실패한 경우
        """
        pass
    
    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        num_predict: Optional[int] = None
    ) -> Generator[str, None, None]:
        """
        LLM을 사용하여 텍스트를 스트리밍 방식으로 생성합니다.
        
        Args:
            prompt: 입력 프롬프트
            temperature: 샘플링 온도
            top_p: Top-p 샘플링 파라미터
            num_predict: 생성할 최대 토큰 수
            
        Yields:
            생성된 텍스트 청크
            
        Raises:
            RuntimeError: 생성이 실패한 경우
        """
        pass
    
    @abstractmethod
    def check_model_available(self, model: str) -> bool:
        """
        모델 사용 가능 여부를 확인합니다.
        
        Args:
            model: 모델명
            
        Returns:
            모델이 사용 가능하면 True, 아니면 False
        """
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """
        사용 가능한 모델 목록을 가져옵니다.
        
        Returns:
            모델명 목록
        """
        pass


# ============================================================================
# 팩토리 함수
# ============================================================================

# 전역 클라이언트 인스턴스
_client: Optional[LLMClient] = None


def get_llm_client(backend_type: Optional[str] = None) -> LLMClient:
    """
    LLM 클라이언트 인스턴스를 가져오거나 생성합니다.
    
    Args:
        backend_type: 백엔드 타입 ("ollama" 또는 "lmstudio")
                     None이면 config에서 가져옴
    
    Returns:
        LLMClient 인스턴스
        
    Raises:
        ValueError: 지원하지 않는 백엔드 타입
    """
    global _client
    
    from .config import config
    
    backend_type = backend_type or config.LLM_BACKEND.BACKEND_TYPE
    
    if _client is None:
        if backend_type == "ollama":
            from .ollama_client import OllamaClient
            _client = OllamaClient()
        elif backend_type == "lmstudio":
            from .lm_studio_client import LMStudioClient
            _client = LMStudioClient()
        else:
            raise ValueError(
                f"지원하지 않는 LLM 백엔드: {backend_type}. "
                "'ollama' 또는 'lmstudio'를 사용하세요."
            )
    
    return _client


def reset_llm_client():
    """전역 LLM 클라이언트를 리셋합니다. (테스트용)"""
    global _client
    _client = None
