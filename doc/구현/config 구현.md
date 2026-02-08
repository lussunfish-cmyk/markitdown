# 
## 개요

애플리케이션 전체의 설정을 중앙에서 관리하는 모듈입니다. 환경 변수(`os.getenv`)를 통해 설정을 주입받을 수 있으며, 기능별로 설정 클래스를 분리하여 관리합니다.
## 파일 경로

```
markitdown/app/config.py
```

## 주요 설정 클래스

- **AplamaConfig**: Ollama 서버 URL, 모델명(LLM, Embedding), 타임아웃 설정.
- **VectorStoreConfig**: 벡터 저장소 타입(Chroma), 경로, 컬렉션 이름 설정.
- **RetrieverConfig**: 검색 관련 설정 (Hybrid Alpha, RRF K, Reranker 모델 등).
- **RAGConfig**: RAG 파이프라인 설정 (Top-K, Temperature, Max Tokens, Query Rewriting 여부).
- **ConversionConfig**: 파일 변환 입출력 경로, 지원 포맷, LibreOffice 설정.
- **IndexingConfig**: 인덱싱 대상 경로, 상태 파일 경로.
- **BatchConfig**: 배치 처리 상태 저장 경로, 배치 크기.
- **PromptConfig**: 시스템 프롬프트, 사용자 프롬프트 템플릿 정의.

## 특징
- **검증 로직**: `validate_config` 함수를 통해 설정값의 유효성(예: alpha 범위 0~1)을 애플리케이션 시작 시 검사합니다.
- **환경 변수 우선**: Docker 환경 등에서 유연하게 설정을 변경할 수 