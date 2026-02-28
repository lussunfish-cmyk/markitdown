"""
LM Studio API 클라이언트 구현.

LM Studio는 LLM 생성용으로만 사용합니다.
임베딩은 별도의 OpenAI 호환 임베딩 서비스를 사용합니다 (localhost:8001 등).
"""

import logging
import json
from typing import List, Optional, Generator
import requests  # type: ignore

from .config import config
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class LMStudioClient(LLMClient):
    """LM Studio API와 상호작용하는 클라이언트."""
    
    def __init__(self):
        """LM Studio 클라이언트를 초기화합니다."""
        self.base_url = config.LMSTUDIO.BASE_URL.rstrip("/")
        self.embedding_service_url = config.LMSTUDIO.EMBEDDING_SERVICE_BASE_URL.rstrip("/")
        self.embedding_model_name = config.LMSTUDIO.EMBEDDING_MODEL
        self.llm_model = config.LMSTUDIO.LLM_MODEL
        self.timeout = config.LMSTUDIO.REQUEST_TIMEOUT
        self.max_retries = config.LMSTUDIO.MAX_RETRIES
        self.retry_delay = config.LMSTUDIO.RETRY_DELAY
        self.embedding_service_available = False  # 임베딩 서비스 가용 여부 플래그
        
        # LM Studio 연결 확인 (LLM 생성용)
        self._verify_connection()
        
        # 임베딩 서비스 연결 확인 (선택적, 실패해도 API는 계속 실행)
        self._verify_embedding_service_connection()
        
        if self.embedding_service_available:
            logger.info(f"✓ LM Studio 클라이언트 초기화됨 (LLM: {self.llm_model}, 임베딩 서비스: {self.embedding_service_url})")
        else:
            logger.info(f"✓ LM Studio 클라이언트 초기화됨 (LLM: {self.llm_model}, 임베딩 서비스: 미사용)")
    
    def _verify_connection(self) -> None:
        """LM Studio 서버 연결을 확인합니다."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            response.raise_for_status()
            logger.info(f"✓ LM Studio 서버 연결됨: {self.base_url}")
        except Exception as e:
            logger.error(f"✗ LM Studio 연결 에러: {str(e)}")
            raise RuntimeError(f"LM Studio 서버 연결 실패: {str(e)}")
    
    def _verify_embedding_service_connection(self) -> None:
        """임베딩 서비스 연결을 확인합니다 (선택적)."""
        try:
            # 먼저 /v1/models 엔드포인트 시도
            try:
                response = requests.get(f"{self.embedding_service_url}/v1/models", timeout=5)
                response.raise_for_status()
            except Exception:
                # /v1/models가 없으면, 간단한 임베딩 요청으로 서비스 확인
                payload = {
                    "input": ["test"],
                    "model": self.embedding_model_name
                }
                response = requests.post(
                    f"{self.embedding_service_url}/v1/embeddings",
                    json=payload,
                    timeout=5
                )
                response.raise_for_status()
            
            logger.info(f"✓ 임베딩 서비스 연결됨: {self.embedding_service_url}")
            self.embedding_service_available = True
        except Exception as e:
            logger.warning(f"⚠ 임베딩 서비스를 사용할 수 없습니다: {str(e)}")
            logger.warning(f"   인덱싱/검색 기능을 사용하려면 {self.embedding_service_url}에서 임베딩 서비스가 실행 중이어야 합니다")
            self.embedding_service_available = False
    
    def embed(self, text: str) -> List[float]:
        """
        텍스트를 위한 임베딩을 생성합니다.
        별도의 임베딩 서비스(OpenAI 호환 API)를 사용합니다.
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            실수 목록으로 된 임베딩 벡터
            
        Raises:
            RuntimeError: 임베딩 생성이 실패한 경우
        """
        if not self.embedding_service_available:
            raise RuntimeError(
                f"임베딩 서비스를 사용할 수 없습니다. "
                f"{self.embedding_service_url}에서 OpenAI 호환 임베딩 서비스가 실행 중인지 확인하세요."
            )
        
        if not text or not text.strip():
            raise ValueError("텍스트는 비어있을 수 없습니다")
        
        try:
            # 임베딩 서비스 API 호출 (OpenAI 호환)
            payload = {
                "input": [text],
                "model": self.embedding_model_name
            }
            
            response = requests.post(
                f"{self.embedding_service_url}/v1/embeddings",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            if "data" not in data or len(data["data"]) == 0:
                raise RuntimeError("임베딩 응답이 없습니다")
            
            # OpenAI 호환 형식: data[0]["embedding"]
            embedding_list = data["data"][0]["embedding"]
            
            return embedding_list
        
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"임베딩 생성 에러: {str(e)}")
            raise RuntimeError(f"임베딩 생성 실패: {str(e)}")
    
    def embed_batch(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        """
        여러 텍스트에 대한 임베딩을 생성합니다.
        임베딩 서비스에 배치로 전송합니다.
        
        Args:
            texts: 임베딩할 텍스트 목록
            show_progress: 진행 상황 기록 여부
            
        Returns:
            임베딩 벡터 목록
        """
        if not self.embedding_service_available:
            raise RuntimeError(
                f"임베딩 서비스를 사용할 수 없습니다. "
                f"{self.embedding_service_url}에서 OpenAI 호환 임베딩 서비스가 실행 중인지 확인하세요."
            )
        
        if not texts:
            return []
        
        if show_progress:
            logger.info(f"배치 임베딩 시작: {len(texts)}개 텍스트")
        
        try:
            # 임베딩 서비스 API 호출 (OpenAI 호환)
            payload = {
                "input": texts,
                "model": self.embedding_model_name
            }
            
            response = requests.post(
                f"{self.embedding_service_url}/v1/embeddings",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            if "data" not in data:
                raise RuntimeError("임베딩 응답이 없습니다")
            
            # OpenAI 호환 형식: 여러 임베딩 처리
            embeddings_list = []
            for item in data["data"]:
                embeddings_list.append(item["embedding"])
            
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
            # 기본 페이로드 - 필수 필드만
            payload = {
                "model": self.llm_model,
                "input": prompt
            }
            
            logger.info(f"LM Studio 요청 페이로드: {payload}")
            
            response = requests.post(
                f"{self.base_url}/api/v1/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            logger.info(f"LM Studio 전체 응답: {data}")
            
            # 에러 응답 확인
            if "error" in data:
                error_msg = data.get("error", "Unknown error")
                logger.error(f"LM Studio 에러: {error_msg}")
                raise RuntimeError(f"LM Studio 에러: {error_msg}")
            
            if "output" not in data or len(data["output"]) == 0:
                logger.error(f"LM Studio 응답에 output이 없습니다. 응답: {data}")
                raise RuntimeError("생성 결과에 응답이 없습니다")
            
            # output 배열에서 type이 "message"인 항목의 content 추출
            for output in data["output"]:
                if isinstance(output, dict) and output.get("type") == "message":
                    content = output.get("content", "").strip()
                    if content:
                        return content
            
            # message 타입이 없으면 첫 번째 output 사용
            if isinstance(data["output"][0], dict):
                content = data["output"][0].get("content", "").strip()
                if content:
                    return content
            
            raise RuntimeError(f"응답 형식 오류: {data}")
        
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
            # 기본 페이로드 - 필수 필드만
            payload = {
                "model": self.llm_model,
                "input": prompt
            }
            
            logger.info(f"LM Studio 스트리밍 요청 페이로드: {payload}")
            
            # LM Studio API는 스트리밍을 지원하지 않으므로 일반 요청 사용
            response = requests.post(
                f"{self.base_url}/api/v1/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            logger.info(f"LM Studio 스트리밍 전체 응답: {data}")
            
            # 에러 응답 확인
            if "error" in data:
                error_msg = data.get("error", "Unknown error")
                logger.error(f"LM Studio 에러: {error_msg}")
                raise RuntimeError(f"LM Studio 에러: {error_msg}")
            
            if "output" not in data or len(data["output"]) == 0:
                logger.error(f"LM Studio 응답에 output이 없습니다. 응답: {data}")
                raise RuntimeError("생성 결과에 응답이 없습니다")
            
            # output 배열에서 type이 "message"인 항목의 content 추출
            content = ""
            for output in data["output"]:
                if isinstance(output, dict) and output.get("type") == "message":
                    content = output.get("content", "").strip()
                    break
            
            if not content and isinstance(data["output"][0], dict):
                content = data["output"][0].get("content", "").strip()
            
            # 전체 텍스트를 한 번에 yield (스트리밍 시뮬레이션)
            if content:
                yield content
        
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
