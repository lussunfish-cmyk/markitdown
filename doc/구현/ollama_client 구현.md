# Ollama Client 구현

## 개요

로컬 LLM 실행 도구인 Ollama 서버와의 통신을 담당하는 클라이언트 모듈입니다. 텍스트 임베딩 생성 및 LLM 답변 생성 기능을 제공합니다.

## 파일 경로

```
markitdown/app/ollama_client.py
```

## 주요 클래스

### OllamaClient
`ollama` Python 라이브러리를 래핑하여 편의 기능을 제공합니다.

- **초기화**: `config.py`에서 설정된 URL 및 모델 정보를 사용하여 클라이언트를 설정하고 연결을 확인합니다.
- **`embed(text)`**: 텍스트를 입력받아 임베딩 벡터(float 리스트)를 반환합니다.
- **`embed_batch(texts)`**: 여러 텍스트에 대한 임베딩을 일괄 생성합니다.
- **`generate(prompt)`**: 프롬프트를 입력받아 LLM의 텍스트 응답을 반환합니다. Temperature, Top-P 등의 파라미터를 설정할 수 있습니다.
- **`stream_generate(prompt)`**: LLM의 응답을 스트리밍(Generator) 형태로 반환하여 실시간 출력을 지원합니다.
- **모델 관리**: `list_models`, `check_model_available` 등을 통해 사용 가능한 모델을 확인합니다.

## 설정
`config.py`의 `OllamaConfig`를 통해 모델명(`gemma2`, `mxbai-embed-large` 등)과 타임아웃 등을 설정합니다.