# Batch Manager 구현

## 개요

대량의 파일 변환 및 처리 작업을 효율적으로 관리하기 위한 모듈입니다. 작업을 작은 배치 단위로 나누어 처리하고, 진행 상황을 파일(JSON)로 저장하여 중단 시 복구(Resume)가 가능하도록 합니다.

## 파일 경로

```
markitdown/app/batch_manager.py
```

## 주요 클래스

### BatchStateManager
배치 작업의 상태를 관리하는 핵심 클래스입니다.

- **상태 저장**: 각 배치 작업의 상태를 `batch_state/` 디렉토리에 JSON 파일로 저장합니다.
- **배치 생성 (`create_batch`)**: 전체 파일 목록을 받아 지정된 크기(`batch_size`)로 나누어 배치 작업을 생성합니다.
- **상태 업데이트**:
  - `update_file_status`: 개별 파일의 처리 결과(성공/실패, 소요 시간 등)를 업데이트합니다.
  - `update_batch_status`: 배치 그룹(Batch Info)의 상태를 업데이트합니다.
- **진행률 추적**: 전체 작업 대비 완료된 파일 수를 계산하여 진행률(%)을 제공합니다.
- **복구 지원**: `get_next_pending_batch` 등을 통해 처리되지 않은 배치를 찾아 작업을 재개할 수 있습니다.

## 데이터 구조
`schemas.py`의 `BatchJobResponse`, `BatchInfo`, `BatchFileStatus` 모델을 사용하여 상태 데이터를 구조화합니다.