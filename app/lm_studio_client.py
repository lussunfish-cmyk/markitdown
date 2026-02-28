"""
LM Studio API 클라이언트 구현.

LM Studio는 LLM 생성용으로만 사용하고,
임베딩은 mlx-embeddings 라이브러리를 별도로 사용합니다.
(LM Studio는 BERT 계열 임베딩 모델을 지원하지 않음)
"""

import logging
import json
import os
from pathlib import Path
import requests  # type: ignore
from typing import List, Optional, Generator

from .config import config
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

# mlx-embeddings 임포트 (optional)
try:
    from mlx_embeddings import load, generate  # type: ignore
    MLX_EMBEDDINGS_AVAILABLE = True
except ImportError:
    MLX_EMBEDDINGS_AVAILABLE = False
    logger.warning("mlx-embeddings가 설치되지 않았습니다. 임베딩 기능을 사용하려면 'pip install mlx-embeddings'를 실행하세요.")
    # 타입 체커를 위한 더미 정의
    load = None  # type: ignore
    generate = None  # type: ignore


class LMStudioClient(LLMClient):
    """LM Studio API와 상호작용하는 클라이언트."""
    
    def __init__(self):
        """LM Studio 클라이언트를 초기화합니다."""
        self.base_url = config.LMSTUDIO.BASE_URL.rstrip("/")
        self.embedding_model_name = config.LMSTUDIO.EMBEDDING_MODEL
        self.llm_model = config.LMSTUDIO.LLM_MODEL
        self.timeout = config.LMSTUDIO.REQUEST_TIMEOUT
        self.max_retries = config.LMSTUDIO.MAX_RETRIES
        self.retry_delay = config.LMSTUDIO.RETRY_DELAY
        
        # 임베딩 모델 경로 확장 (~ 및 환경 변수 처리)
        self.embedding_model_path = self._expand_model_path(self.embedding_model_name)
        
        # LM Studio 연결 확인 (LLM 생성용)
        self._verify_connection()
        
        # mlx-embeddings 모델 로드 (임베딩용)
        self.embedding_model = None
        self.embedding_processor = None
        if MLX_EMBEDDINGS_AVAILABLE:
            try:
                logger.info(f"mlx-embeddings 모델 로드 중: {self.embedding_model_path}")
                self.embedding_model, self.embedding_processor = load(self.embedding_model_path)  # type: ignore
                logger.info(f"✓ mlx-embeddings 모델 로드 완료: {self.embedding_model_path}")
            except Exception as e:
                logger.error(f"✗ mlx-embeddings 모델 로드 실패: {str(e)}")
                logger.warning("임베딩 기능이 비활성화됩니다. LLM 생성만 사용 가능합니다.")
        
        logger.info(f"✓ LM Studio 클라이언트 초기화됨 (LLM: {self.llm_model})")
    
    def _expand_model_path(self, model_path: str) -> str:
        """
        모델 경로를 확장하고 검증합니다.
        
        Args:
            model_path: 모델 경로 또는 Hugging Face 모델명
            
        Returns:
            확장된 경로 또는 원본 모델명
        """
        # ~ (홈 디렉토리) 확장
        expanded_path = os.path.expanduser(model_path)
        
        # 로컬 경로인 경우 (절대 경로 또는 상대 경로로 시작)
        if expanded_path.startswith('/') or expanded_path.startswith('./') or expanded_path.startswith('../'):
            path_obj = Path(expanded_path)
            
            # 경로가 존재하는지 확인
            if path_obj.exists():
                logger.info(f"오프라인 모델 경로 확인됨: {expanded_path}")
                return str(path_obj.absolute())
            else:
                logger.warning(f"로컬 경로를 찾을 수 없습니다: {expanded_path}")
                logger.warning("Hugging Face에서 온라인 다운로드를 시도합니다.")
                # 경로가 존재하지 않으면 원본 경로 반환 (온라인 다운로드 시도)
                return model_path
        
        # Hugging Face 모델명인 경우 (예: mlx-community/mxbai-embed-large-v1)
        logger.info(f"Hugging Face 모델명 감지: {model_path}")
        return model_path
    
    def _verify_connection(self) -> None:
        """LM Studio 서버 연결을 확인합니다."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            response.raise_for_status()
            logger.info(f"✓ LM Studio 서버 연결됨: {self.base_url}")
        except Exception as e:
            logger.error(f"✗ LM Studio 연결 에러: {str(e)}")
            raise RuntimeError(f"LM Studio 서버 연결 실패: {str(e)}")
    
    def embed(self, text: str) -> List[float]:
        """
        텍스트를 위한 임베딩을 생성합니다.
        mlx-embeddings 라이브러리를 사용합니다 (LM Studio API 아님).
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            실수 목록으로 된 임베딩 벡터
            
        Raises:
            RuntimeError: 임베딩 생성이 실패한 경우
        """
        if not text or not text.strip():
            raise ValueError("텍스트는 비어있을 수 없습니다")
        
        if not MLX_EMBEDDINGS_AVAILABLE or self.embedding_model is None:
            raise RuntimeError(
                "mlx-embeddings가 설치되지 않았거나 모델 로드에 실패했습니다. "
                "'pip install mlx-embeddings'를 실행하고 다시 시도하세요."
            )
        
        try:
            # mlx-embeddings를 사용하여 임베딩 생성
            embeddings = generate(  # type: ignore
                self.embedding_model,
                self.embedding_processor,
                texts=[text],
                normalize=True
            )
            
            # numpy array를 list로 변환
            if hasattr(embeddings, 'tolist'):
                embedding_list = embeddings[0].tolist()
            else:
                embedding_list = list(embeddings[0])
            
            return embedding_list
        
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"임베딩 생성 에러: {str(e)}")
            raise RuntimeError(f"임베딩 생성 실패: {str(e)}")
    
    def embed_batch(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        """
        여러 텍스트에 대한 임베딩을 생성합니다.
        mlx-embeddings 라이브러리를 사용하여 배치로 처리합니다.
        
        Args:
            texts: 임베딩할 텍스트 목록
            show_progress: 진행 상황 기록 여부
            
        Returns:
            임베딩 벡터 목록
        """
        if not MLX_EMBEDDINGS_AVAILABLE or self.embedding_model is None:
            raise RuntimeError(
                "mlx-embeddings가 설치되지 않았거나 모델 로드에 실패했습니다. "
                "'pip install mlx-embeddings'를 실행하고 다시 시도하세요."
            )
        
        if show_progress:
            logger.info(f"배치 임베딩 시작: {len(texts)}개 텍스트")
        
        try:
            # mlx-embeddings로 배치 처리 (훨씬 빠름)
            embeddings = generate(  # type: ignore
                self.embedding_model,
                self.embedding_processor,
                texts=texts,
                normalize=True
            )
            
            # numpy array를 list로 변환
            if hasattr(embeddings, 'tolist'):
                embeddings_list = embeddings.tolist()
            else:
                embeddings_list = [list(emb) for emb in embeddings]
            
            if show_progress:
                logger.info(f"배치 임베딩 완료: {len(embeddings_list)}개")
            
            return embeddings_list
        
        except Exception as e:
            logger.error(f"배치 임베딩 생성 에러: {str(e)}")
            # 실패 시 개별 처리로 fallback (진행 상황 표시)
            logger.warning("개별 임베딩 처리로 전환합니다...")
            embeddings = []
            
            for i, text in enumerate(texts):
                if show_progress and (i % 10 == 0 or i == len(texts) - 1):
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
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature or config.RAG.TEMPERATURE,
                "max_tokens": num_predict or config.RAG.MAX_TOKENS,
                "stream": False
            }
            
            if top_p is not None:
                payload["top_p"] = top_p
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            if "choices" not in data or len(data["choices"]) == 0:
                raise RuntimeError("생성 결과에 응답이 없습니다")
            
            return data["choices"][0]["message"]["content"].strip()
        
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"텍스트 생성 에러: {str(e)}")
            raise RuntimeError(f"텍스트 생성 실패: {str(e)}")
    
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
        if not prompt or not prompt.strip():
            raise ValueError("프롬프트는 비어있을 수 없습니다")
        
        try:
            payload = {
                "model": self.llm_model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature or config.RAG.TEMPERATURE,
                "max_tokens": num_predict or config.RAG.MAX_TOKENS,
                "stream": True
            }
            
            if top_p is not None:
                payload["top_p"] = top_p
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        if data_str.strip() == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue
        
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"스트리밍 생성 에러: {str(e)}")
            raise RuntimeError(f"스트리밍 생성 실패: {str(e)}")
    
    def check_model_available(self, model: str) -> bool:
        """
        모델 사용 가능 여부를 확인합니다.
        
        Args:
            model: 모델명
            
        Returns:
            모델이 사용 가능하면 True, 아니면 False
        """
        try:
            models = self.list_models()
            return any(model in m for m in models)
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
            response = requests.get(
                f"{self.base_url}/v1/models",
                timeout=5
            )
            response.raise_for_status()
            
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.error(f"모델 목록 조회 에러: {str(e)}")
            return []
