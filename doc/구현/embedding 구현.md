# embedding.py 구현

## 개요

텍스트 문서(특히 Markdown)를 의미 있는 단위인 **청크(Chunk)**로 분할하고, Ollama API를 통해 **임베딩 벡터(Embedding Vector)**를 생성하는 핵심 모듈입니다. RAG 성능에 결정적인 영향을 미치는 전처리 과정을 담당합니다.

## 파일 경로

```
markitdown/app/embedding.py
```

## 주요 클래스

### 1. TextChunker

범용 텍스트 분할 클래스입니다.

- **기능**: 주어진 `chunk_size`와 `chunk_overlap`에 맞춰 텍스트를 자릅니다.
- **알고리즘**: `\n\n`, `\n`, `.`, ` ` 등의 구분자(Separator) 우선순위를 사용하여, 의미 단위가 깨지지 않도록 재귀적으로 분할합니다 (`_split_text_recursive`).
- **병합 로직**: 분할된 작은 조각들을 최대 크기에 근접하도록 다시 병합합니다 (`_merge_chunks`).
- **강제 분할**: 어떤 구분자로도 나눌 수 없는 긴 문자열은 강제로 자릅니다.

### 2. MarkdownChunker (TextChunker 상속)

Markdown 문서 구조에 특화된 청커입니다.

- **헤더 인식**: `#`, `##` 등의 헤더를 인식하여 문서를 섹션 단위로 파싱합니다 (`_extract_sections`).
- **컨텍스트 유지**: 분할된 청크가 어떤 섹션에 속하는지 알 수 있도록 상위 헤더 정보를 유지합니다.
  - 예: `[1. 서론 > 1.1 배경]\n\n실제 내용...`
- **섹션 보존**: 가급적 같은 섹션의 내용은 같은 청크에 담거나, 섹션 단위로 분할합니다.

### 3. DocumentEmbedder

문서 파일을 읽어 청크로 나누고 임베딩까지 수행하는 통합 클래스입니다.

- **초기화**: `use_markdown_chunker=True`일 경우 `MarkdownChunker`를 사용합니다.
- **`embed_document` 메서드**:
  1. `chunker.split_text`로 텍스트 분할.
  2. `ollama_client.embed`로 각 청크의 벡터 생성.
  3. 청크별 고유 ID 생성 (Source + Index 해시).
  4. 메타데이터(Source, Chunk ID, Created At) 생성.
  5. `DocumentChunk` 객체 리스트 반환.
- **배치 처리**: `embed_batch` 메서드로 여러 문서를 한 번에 처리합니다.

## 데이터 흐름

1. **입력**: Markdown 파일 또는 텍스트.
2. **청킹**: `MarkdownChunker`가 헤더 구조를 분석하여 섹션별로 나누고, 크기 제한에 맞춰 세부 분할. 이때 상위 헤더 정보를 텍스트 앞단에 주입(Injection)하여 문맥 소실 방지.
3. **임베딩**: `OllamaClient`를 통해 분할된 각 텍스트 조각을 벡터화.
4. **결과**: `DocumentChunk` 리스트 (텍스트 + 벡터 + 메타데이터).
