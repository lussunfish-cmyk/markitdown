# ollama_client.py 구현

## 개요

Ollama 서버와 통신하여 임베딩 생성 및 LLM 텍스트 생성 기능을 제공하는 클라이언트 모듈입니다. 연결 관리, 재시도 로직, 에러 핸들링을 포함합니다.

## 파일 경로

```
markitdown/app/ollama_client.py
```

## 주요 구성 요소

### 1. OllamaClient 클래스

Ollama 서버와의 모든 상호작용을 관리하는 메인 클래스입니다.

```python
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
```

**초기화 과정:**
1. config에서 설정 로드
2. HTTP 세션 생성 (재시도 전략 포함)
3. Ollama 서버 연결 확인

### 2. 세션 관리

#### _create_session()

재시도 전략을 포함한 HTTP 세션을 생성합니다.

```python
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
```

**재시도 전략:**
- 총 재시도 횟수: `max_retries` (기본 3회)
- 백오프 팩터: 1 (1초, 2초, 4초...)
- 재시도 대상 상태 코드: 429, 500, 502, 503, 504
- 허용 메서드: GET, POST

#### _verify_connection()

Ollama 서버 연결을 확인합니다.

```python
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
```

**연결 확인 과정:**
1. `/api/tags` 엔드포인트 호출
2. 응답 상태 확인
3. 성공 시 로그 출력
4. 실패 시 RuntimeError 발생

### 3. API 호출 (재시도 로직)

#### _call_with_retry()

재시도 로직을 포함한 Ollama API 호출 메서드입니다.

```python
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
```

**재시도 시나리오:**

1. **Timeout 발생**
   - 지정된 시간 내 응답 없음
   - `retry_delay` 후 재시도
   - 최대 `max_retries`회까지 재시도

2. **Connection Error**
   - 네트워크 연결 실패
   - `retry_delay` 후 재시도

3. **HTTP Error (4xx/5xx)**
   - HTTP 에러는 재시도하지 않음 (즉시 실패)
   - 클라이언트 오류는 재시도해도 해결 안 됨

**에러 처리:**
```python
except requests.Timeout:
    last_error = f"요청 타임아웃 ({timeout}초)"
    if attempt < self.max_retries - 1:
        logger.warning(f"재시도 {attempt + 1}/{self.max_retries}")
        time.sleep(self.retry_delay)

except requests.HTTPError as e:
    # HTTP 에러는 재시도하지 않음
    raise RuntimeError(f"HTTP 에러 {e.response.status_code}")
```

### 4. 임베딩 생성

#### embed()

단일 텍스트의 임베딩을 생성합니다.

```python
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
```

**사용 예시:**
```python
client = OllamaClient()
text = "이것은 테스트 문장입니다."
embedding = client.embed(text)
print(f"임베딩 차원: {len(embedding)}")  # 768 (nomic-embed-text)
```

#### embed_batch()

여러 텍스트의 임베딩을 배치로 생성합니다.

```python
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
```

**특징:**
- 실패한 텍스트는 0 벡터로 대체
- 진행 상황 로깅 옵션
- 일부 실패해도 전체 처리 계속

### 5. 텍스트 생성

#### generate()

LLM을 사용하여 텍스트를 생성합니다.

```python
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
        
        # ... (top_p, num_predict 설정)
        
        response = self._call_with_retry("generate", payload)
        
        if "response" not in response:
            raise RuntimeError("생성 결과에 응답이 없습니다")
        
        return response["response"].strip()
```

**사용 예시:**
```python
client = OllamaClient()

# 기본 설정 사용
answer = client.generate("5G의 장점은 무엇인가?")

# 커스텀 파라미터
answer = client.generate(
    prompt="설명해주세요",
    temperature=0.5,  # 더 결정적
    max_tokens=200
)
```

### 6. 모델 관리

#### check_model_available()

모델 사용 가능 여부를 확인합니다.

```python
def check_model_available(self, model: str) -> bool:
    """
    Ollama에서 모델 사용 가능 여부를 확인합니다.
    
    Args:
        model: 모델명
        
    Returns:
        모델이 사용 가능하면 True, 아니면 False
    """
```

#### list_models()

사용 가능한 모델 목록을 반환합니다.

```python
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
```

#### get_model_info()

특정 모델의 상세 정보를 반환합니다.

```python
def get_model_info(self, model: str) -> Optional[dict]:
    """
    모델에 대한 상세 정보를 가져옵니다.
    
    Args:
        model: 모델명
        
    Returns:
        모델 정보 딕셔너리 또는 없으면 None
    """
```

### 7. 싱글톤 패턴

전역 클라이언트 인스턴스를 관리합니다.

```python
# 전역 클라이언트 인스턴스
_client: Optional[OllamaClient] = None

def get_ollama_client() -> OllamaClient:
    """Ollama 클라이언트 인스턴스를 가져오거나 생성합니다."""
    global _client
    
    if _client is None:
        _client = OllamaClient()
    
    return _client
```

**사용 예시:**
```python
from ollama_client import get_ollama_client

# 어디서든 동일한 인스턴스 사용
client = get_ollama_client()
embedding = client.embed("텍스트")
```

## 사용 예시

### 기본 사용

```python
from ollama_client import get_ollama_client

# 클라이언트 가져오기
client = get_ollama_client()

# 임베딩 생성
text = "5G 기술의 특징"
embedding = client.embed(text)
print(f"차원: {len(embedding)}")

# 텍스트 생성
prompt = "5G의 주요 장점 3가지를 설명해주세요."
answer = client.generate(prompt)
print(answer)
```

### 배치 처리

```python
# 여러 문서 임베딩
documents = [
    "첫 번째 문서 내용",
    "두 번째 문서 내용",
    "세 번째 문서 내용"
]

embeddings = client.embed_batch(documents, show_progress=True)
print(f"총 {len(embeddings)}개 임베딩 생성")
```

### 모델 확인

```python
# 사용 가능한 모델 확인
models = client.list_models()
print("사용 가능한 모델:", models)

# 특정 모델 확인
if client.check_model_available("gemma2"):
    print("gemma2 사용 가능")
else:
    print("gemma2 설치 필요")
```

### 에러 처리

```python
try:
    embedding = client.embed("텍스트")
except ValueError as e:
    print(f"입력 오류: {e}")
except RuntimeError as e:
    print(f"Ollama 오류: {e}")
```

## 에러 처리 전략

### 1. 입력 검증

```python
if not text or not text.strip():
    raise ValueError("텍스트는 비어있을 수 없습니다")
```

### 2. 연결 오류

```python
except requests.ConnectionError:
    raise RuntimeError("Ollama 서버 연결 불가")
```

### 3. 타임아웃

```python
except requests.Timeout:
    # 재시도 로직 실행
    if attempt < max_retries - 1:
        time.sleep(retry_delay)
```

### 4. HTTP 에러

```python
except requests.HTTPError as e:
    # 재시도하지 않고 즉시 실패
    raise RuntimeError(f"HTTP 에러 {e.response.status_code}")
```

## 성능 최적화

### 1. 연결 풀링

```python
# requests.Session으로 연결 재사용
self.session = requests.Session()
```

### 2. 타임아웃 설정

```python
# 무한 대기 방지
response = self.session.get(url, timeout=self.timeout)
```

### 3. 재시도 전략

```python
# 일시적 오류에 대한 자동 재시도
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
```

## 설계 원칙

1. **재시도 로직**: 네트워크 문제에 대한 자동 재시도
2. **에러 핸들링**: 명확한 에러 메시지와 적절한 예외 타입
3. **싱글톤 패턴**: 전역적으로 하나의 클라이언트 인스턴스만 사용
4.**로깅**: 중요한 이벤트와 에러 로깅
5. **타입 안전성**: Type hints를 통한 명확한 인터페이스
6. **설정 분리**: config 모듈을 통한 중앙화된 설정 관리

## 확장 가이드

### 새로운 Ollama API 기능 추가

```python
def new_ollama_feature(self, param: str) -> dict:
    """새로운 Ollama 기능."""
    payload = {
        "model": self.llm_model,
        "param": param
    }
    
    response = self._call_with_retry("new_endpoint", payload)
    return response
```

### 커스텀 재시도 로직

```python
def custom_call(self, endpoint: str, payload: dict, max_attempts: int = 5):
    """커스텀 재시도 횟수로 API 호출."""
    # 구현...
```
