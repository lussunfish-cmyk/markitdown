# Embedding 구현

## 개요

텍스트 문서를 작은 청크(Chunk)로 분할하고, 이를 임베딩 벡터로 변환하는 모듈입니다. 특히 마크다운 문서의 구조(헤더)를 인식하여 의미 단위로 분할하는 기능을 제공합니다.

## 파일 경로

```
markitdown/app/embedding.py
```

## 주요 클래스

### 1. TextChunker
기본적인 텍스트 분할 클래스입니다.
- 지정된 구분자(`\n\n`, `\n`, `. ` 등)를 사용하여 재귀적으로 텍스트를 분할합니다.
- `chunk_size`와 `chunk_overlap` 설정을 준수합니다.

### 2. MarkdownChunker (extends TextChunker)
마크다운 문법을 이해하는 향상된 청커입니다.
- **헤더 기반 분할**: `#`, `##` 등의 헤더를 기준으로 섹션을 나눕니다.
- **컨텍스트 주입 (Context Injection)**: 분할된 각 청크의 앞부분에 상위 헤더 정보(예: `[1. 서론 > 1.1 배경]`)를 자동으로 추가합니다. 이를 통해 청크가 독립적으로 존재할 때도 문맥을 유지할 수 있어 검색 정확도가 향상됩니다.

### 3. DocumentEmbedder
문서를 입력받아 청킹하고 임베딩을 생성하는 클래스입니다.
- `embed_document`: 텍스트 내용을 받아 `DocumentChunk` 객체 리스트를 반환합니다.
- `OllamaClient`를 사용하여 실제 임베딩 벡터를 생성합니다.
- 각 청크에 대한 메타데이터(소스, 청크 ID, 섹션 제목 등)를 생성합니다.

## 데이터 흐름

1. 원본 텍스트 입력.
2. `MarkdownChunker`가 텍스트를 섹션 단위로 파싱.
3. 각 섹션에 상위 헤더 컨텍스트 추가.
4. `chunk_size`에 맞춰 텍스트 분할.
5. `OllamaClient`로 각 청크의 임베딩 벡터 생성.
6. `DocumentChunk` 객체(ID, Content, Metadata, Embedding) 생성 및 반환.