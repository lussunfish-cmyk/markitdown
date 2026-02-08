# ollama_client.py 구현

## 개요

이 모듈은 Ollama API 서버와 통신하여 텍스트 임베딩 생성(Embedding) 및 LLM 텍스트 생성(Generate) 기능을 제공하는 클라이언트입니다. Python 공식 `ollama` 라이브러리를 래핑하여 사용하며, 연결 관리, 에러 처리, 편의 기능을 제공합니다.

## 파일 경로

```
markitdown/app/ollama_client.py
```

## 주요 구성 요소

### 1. OllamaClient 클래스

Ollama 서버와의 모든 상호작용을 담당하는 메인 클래스입니다.

**초기화 (`__init__`):**
- `config.py`에서 설정(BASE_URL, 모델명 등)을 로드합니다.
- `ollama.Client` 인스턴스를 생성합니다.
- `_verify_connection()`을 호출하여 서버 연결 상태를 확인하고, 로깅합니다.

### 2. 주요 메서드

#### `embed(text: str) -> List[float]`
- 단일 텍스트에 대한 임베딩 벡터를 생성합니다.
- `ollama.Client.embed` 메서드를 사용합니다.
- 응답 형식을 검증하고 벡터(float 리스트)를 반환합니다.
- 입력값이 비어있거나 API 에러 발생 시 예외를 발생시킵니다.

#### `embed_batch(texts: List[str], show_progress: bool = False) -> List[List[float]]`
- 여러 텍스트를 순차적으로 임베딩합니다.
- 실패 시 해당 항목에 대해 0으로 채워진 벡터(Zero Vector)를 반환하여 프로세스 중단을 방지합니다.

#### `generate(prompt: str, ...) -> str`
- LLM을 사용하여 주어진 프롬프트에 대한 응답을 생성합니다.
- `temperature`, `top_p`, `num_predict` 등의 파라미터를 설정할 수 있습니다.
- 기본값은 `config.RAG` 설정을 따릅니다.

#### `stream_generate(prompt: str, ...)`
- LLM 응답을 스트리밍(Generator) 방식으로 반환합니다.
- 긴 텍스트 생성 시 실시간 응답을 제공하기 위해 사용됩니다.

#### `check_model_available(model: str) -> bool`
- 특정 모델이 Ollama 서버에 설치되어 있는지 확인합니다.
- 모델명의 일부만 일치해도(접두사 매칭 등) True를 반환하도록 구현되어 유연성을 제공합니다.

#### `list_models() -> List[str]`
- 사용 가능한 모든 모델의 이름 목록을 반환합니다.

#### `get_model_info(model: str) -> Optional[dict]`
- 특정 모델의 상세 정보(크기, 수정일 등)를 반환합니다.

### 3. 싱글톤 패턴

#### `get_ollama_client() -> OllamaClient`
- 전역 `_client` 인스턴스를 관리합니다.
- 클라이언트가 한 번만 초기화되도록 보장합니다.

## 의존성

- **ollama**: Python 공식 Ollama 클라이언트 라이브러리
- **config**: 설정 정보 로드 (URL, 모델명 등)

## 데이터 흐름

1. App 요청 (`embed` / `generate`)
2. `OllamaClient` 메서드 호출
3. HTTP Request -> Ollama Server (localhost:11434)
4. GPU/CPU 모델 추론
5. HTTP Response -> `OllamaClient` -> 결과 파싱 -> App 반환
