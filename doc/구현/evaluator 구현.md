# Evaluator 구현

## 개요

RAG 시스템의 검색 성능을 정량적으로 평가하기 위한 모듈입니다. Ragas 라이브러리를 사용하여 테스트셋을 자동으로 생성하고, 생성된 테스트셋을 바탕으로 검색 정확도를 측정합니다.

## 파일 경로

```
markitdown/app/evaluator.py
```

## 주요 기능

### 1. 테스트셋 생성 (`generate_testset_logic`)
- **Ragas 연동**: `TestsetGenerator`를 사용하여 문서로부터 질문(Question), 정답(Ground Truth), 컨텍스트(Context) 쌍을 생성합니다.
- **청킹 일치**: 인덱싱 시와 동일한 `MarkdownChunker`를 사용하여 테스트셋을 생성함으로써, 평가 시 데이터 불일치 문제를 방지합니다.
- **질문 유형**: Simple, Reasoning, Multi-context 등 다양한 유형의 질문을 생성할 수 있습니다. (현재 설정은 Simple 비중을 높여 할루시네이션 방지)

### 2. 검색 성능 평가 (`evaluate_retrieval_logic`)
- 생성된 테스트셋의 질문을 RAG 시스템(Retriever)에 입력하여 문서를 검색합니다.
- 검색된 문서와 정답(Ground Truth)을 비교하여 다음 지표를 계산합니다:
  - **Recall**: 정답 문서를 얼마나 잘 찾았는가.
  - **Precision**: 검색된 문서 중 정답의 비율.
  - **F1 Score**: Recall과 Precision의 조화 평균.
  - **MRR (Mean Reciprocal Rank)**: 정답 문서가 몇 번째 순위에 등장하는가.
  - **Hit Rate**: 정답 문서를 하나라도 찾았는가.

### 3. 텍스트 정규화 (`normalize_text`)
- 평가의 정확도를 높이기 위해 텍스트의 대소문자 통일, 구두점 제거, 공백 정리 등을 수행합니다.
- `Fuzzy Matching`: `difflib`을 사용하여 미세한 차이가 있어도 유사도가 높으면 정답으로 인정하는 로직이 포함되어 있습니다.

## 스키마
- `schemas_eval.py`에 정의된 `TestsetGenerateRequest`, `EvaluationResponse` 등을 사용합니다.