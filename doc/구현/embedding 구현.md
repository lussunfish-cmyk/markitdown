# embedding.py 구현 정리

이 문서는 embedding.py 구현 및 테스트 구성 내용을 요약한다.

## 구현 범위

- 텍스트 청킹 및 마크다운 구조 인식 청킹
- Ollama 임베딩 생성
- 문서 단위 임베딩 처리 및 배치 처리
- 테스트 실행 시 청크 결과 CSV 저장

## 주요 클래스 및 함수

### TextChunker

- 텍스트를 설정된 크기로 분할
- 구분자 우선순위 기반 재귀 분할
- 청크 병합 및 강제 분할 지원
- 오버랩 적용 분할 지원

### MarkdownChunker

- 마크다운 구조를 고려한 구분자 적용
- 1차 헤더(#) 기준 섹션 분할 후 청킹

### DocumentEmbedder

- 텍스트 청킹 후 임베딩 생성
- DocumentChunk 및 DocumentMetadata 생성
- 배치 임베딩 처리 지원

### helper

- create_embedder: DocumentEmbedder 생성
- chunk_text: 간단한 텍스트 청킹

## 테스트 구성

- test_embedding.py에서 5개 테스트 실행
  - TextChunker 기본 동작
  - MarkdownChunker 구조 인식
  - DocumentEmbedder 임베딩 생성
  - 실제 마크다운 파일 임베딩
  - chunk_text 헬퍼 함수

## CSV 출력

- 테스트 실행 시 청크 결과를 CSV로 저장
- 저장 경로
  - 컨테이너: /app/debug
  - 로컬: ./debug
- docker-compose.yml에 debug 디렉토리 마운트 추가
- .gitignore에 debug/ 추가

## 실행

- 컨테이너 내부 실행 예시
  - docker compose exec -T markitdown-api python test_embedding.py
