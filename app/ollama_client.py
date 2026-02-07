"""
임베딩 및 LLM 상호작용을 위한 Ollama 클라이언트.
"""

import logging
from typing import Optional, List

import ollama

from .config import config

logger = logging.getLogger(__name__)


class OllamaClient:
    """Ollama 서버와 상호작용하는 클라이언트."""
    
    def __init__(self):
        """Ollama 클라이언트를 초기화합니다."""
        self.base_url = config.OLLAMA.BASE_URL.rstrip("/")
        self.embedding_model = config.OLLAMA.EMBEDDING_MODEL
        self.llm_model = config.OLLAMA.LLM_MODEL
        self.timeout = config.OLLAMA.REQUEST_TIMEOUT
        self.max_retries = config.OLLAMA.MAX_RETRIES
        self.retry_delay = config.OLLAMA.RETRY_DELAY
        
        # Ollama 클라이언트 설정
        self.client = ollama.Client(host=self.base_url)
        
        # 연결 확인
        self._verify_connection()
    
    def _verify_connection(self) -> None:
        """Ollama 서버 연결을 확인합니다."""
        try:
            self.client.list()
            logger.info(f"✓ Ollama 서버 연결됨: {self.base_url}")
        except Exception as e:
            logger.error(f"✗ Ollama 연결 에러: {str(e)}")
            raise RuntimeError(f"Ollama 서버 연결 실패: {str(e)}")
    
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
        if not text or not text.strip():
            raise ValueError("텍스트는 비어있을 수 없습니다")
        
        try:
            response = self.client.embed(
                model=self.embedding_model,
                input=text
            )
            
            # embeddings는 리스트의 리스트이므로 첫 번째 요소를 반환
            if not response.get('embeddings') or len(response['embeddings']) == 0:
                raise RuntimeError("Ollama 응답에 임베딩이 없습니다")
            
            embedding = response['embeddings'][0]
            
            if not isinstance(embedding, list):
                raise RuntimeError(f"잘못된 임베딩 형식: {type(embedding)}")
            
            return embedding
        
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"임베딩 생성 에러: {str(e)}")
            raise RuntimeError(f"임베딩 생성 실패: {str(e)}")
    
    def embed_batch(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        """
        여러 텍스트에 대한 임베딩을 생성합니다.
        
        Args:
            texts: 임베딩할 텍스트 목록
            show_progress: 진행 상황 기록 여부
            
        Returns:
            임베딩 벡터 목록
        """
        embeddings = []
        
        for i, text in enumerate(texts):
            if show_progress:
                logger.info(f"임베딩 {i + 1}/{len(texts)}")
            
            try:
                embedding = self.embed(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"텍스트 {i} 임베딩 실패: {str(e)}")
                # 실패 시 빈 임베딩 반환
                embeddings.append([0.0] * config.VECTOR_STORE.EMBEDDING_DIM)
        
        return embeddings
    
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
        if not prompt or not prompt.strip():
            raise ValueError("프롬프트는 비어있을 수 없습니다")
        
        try:
            options = {}
            
            if temperature is not None:
                options["temperature"] = temperature
            else:
                options["temperature"] = config.RAG.TEMPERATURE
            
            if top_p is not None:
                options["top_p"] = top_p
            else:
                options["top_p"] = config.RAG.TOP_P
            
            if num_predict is not None:
                options["num_predict"] = num_predict
            else:
                options["num_predict"] = config.RAG.MAX_TOKENS
            
            response = self.client.generate(
                model=self.llm_model,
                prompt=prompt,
                options=options
            )
            
            if "response" not in response:
                raise RuntimeError("생성 결과에 응답이 없습니다")
            
            return response["response"].strip()
        
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"텍스트 생성 에러: {str(e)}")
            raise RuntimeError(f"텍스트 생성 실패: {str(e)}")
    
    def check_model_available(self, model: str) -> bool:
        """
        Ollama에서 모델 사용 가능 여부를 확인합니다.
        
        Args:
            model: 모델명
            
        Returns:
            모델이 사용 가능하면 True, 아니면 False
        """
        try:
            models = self.client.list()
            model_names = [m.get("model", "") for m in models.get("models", [])]
            
            # 정확한 이름 또는 접두사 매칭
            return any(model in name for name in model_names)
        except Exception as e:
            logger.error(f"모델 가용성 확인 에러: {str(e)}")
            return False
    
    def list_models(self) -> List[str]:
        """
        사용 가능한 모델 목록을 가져옵니다.
        
        Returns:
            모델명 목록
        """
        try:
            response = self.client.list()
            return [m.get("model", "") for m in response.get("models", [])]
        except Exception as e:
            logger.error(f"모델 목록 조회 에러: {str(e)}")
            return []
    
    def get_model_info(self, model: str) -> Optional[dict]:
        """
        모델에 대한 상세 정보를 가져옵니다.
        
        Args:
            model: 모델명
            
        Returns:
            모델 정보 딕셔너리 또는 없으면 None
        """
        try:
            response = self.client.list()
            models = response.get("models", [])
            for m in models:
                if model in m.get("model", ""):
                    return m
            
            return None
        except Exception as e:
            logger.error(f"모델 정보 조회 에러: {str(e)}")
            return None


# 전역 클라이언트 인스턴스
_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Ollama 클라이언트 인스턴스를 가져오거나 생성합니다."""
    global _client
    
    if _client is None:
        _client = OllamaClient()
    
    return _client
