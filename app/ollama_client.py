"""
임베딩 및 LLM 상호작용을 위한 Ollama 클라이언트.
"""

import logging
import time
from typing import Optional, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
        
        # 재시도 전략이 있는 세션 생성
        self.session = self._create_session()
        
        # 연결 확인
        self._verify_connection()
    
    def _create_session(self) -> requests.Session:
        """재시도 전략이 있는 요청 세션을 생성합니다."""
        session = requests.Session()
        
        # 재시도 전략 설정
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _verify_connection(self) -> None:
        """Ollama 서버 연결을 확인합니다."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            response.raise_for_status()
            logger.info(f"✓ Ollama 서버 연결됨: {self.base_url}")
        except requests.ConnectionError:
            logger.error(f"✗ {self.base_url}의 Ollama 서버에 연결할 수 없습니다")
            raise RuntimeError(f"Ollama 서버가 {self.base_url}에서 사용 불가능합니다")
        except Exception as e:
            logger.error(f"✗ Ollama 연결 에러: {str(e)}")
            raise RuntimeError(f"Ollama 서버 연결 실패: {str(e)}")
    
    def _call_with_retry(
        self,
        endpoint: str,
        payload: dict,
        timeout: Optional[int] = None
    ) -> dict:
        """
        재시도 로직을 포함한 Ollama API 호출.
        
        Args:
            endpoint: API 엔드포인트 (기본 URL 제외)
            payload: 요청 페이로드
            timeout: 요청 타임아웃 (초)
            
        Returns:
            API 응답 딕셔너리
            
        Raises:
            RuntimeError: 모든 재시도 후 요청이 실패한 경우
        """
        timeout = timeout or self.timeout
        url = f"{self.base_url}/api/{endpoint}"
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=timeout,
                    stream=False
                )
                response.raise_for_status()
                return response.json()
            
            except requests.Timeout:
                last_error = f"요청 타임아웃 ({timeout}초)"
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"{endpoint}에서 타임아웃, {self.retry_delay}초 후 재시도... "
                        f"(시도 {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(self.retry_delay)
            
            except requests.ConnectionError as e:
                last_error = f"연결 에러: {str(e)}"
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"{endpoint}에서 연결 에러, {self.retry_delay}초 후 재시도... "
                        f"(시도 {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(self.retry_delay)
            
            except requests.HTTPError as e:
                # HTTP 에러는 재시도하지 않음
                raise RuntimeError(
                    f"Ollama HTTP 에러 {e.response.status_code}: {e.response.text}"
                )
            
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Error on {endpoint}, retrying in {self.retry_delay}s... "
                        f"(attempt {attempt + 1}/{self.max_retries}): {str(e)}"
                    )
                    time.sleep(self.retry_delay)
        
        raise RuntimeError(
            f"Failed to call {endpoint} after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )
    
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
            payload = {
                "model": self.embedding_model,
                "prompt": text
            }
            
            response = self._call_with_retry("embed", payload)
            
            if "embedding" not in response:
                raise RuntimeError("Ollama 응답에 임베딩이 없습니다")
            
            embedding = response["embedding"]
            
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
            payload = {
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False
            }
            
            # 선택적 파라미터 추가
            if temperature is not None:
                payload["temperature"] = temperature
            else:
                payload["temperature"] = config.RAG.TEMPERATURE
            
            if top_p is not None:
                payload["top_p"] = top_p
            else:
                payload["top_p"] = config.RAG.TOP_P
            
            if num_predict is not None:
                payload["num_predict"] = num_predict
            else:
                payload["num_predict"] = config.RAG.MAX_TOKENS
            
            response = self._call_with_retry("generate", payload)
            
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
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            response.raise_for_status()
            
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            
            return model in model_names
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
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            response.raise_for_status()
            
            models = response.json().get("models", [])
            return [m.get("name", "") for m in models]
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
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            response.raise_for_status()
            
            models = response.json().get("models", [])
            for m in models:
                if m.get("name", "").startswith(model):
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
