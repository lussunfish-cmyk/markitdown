# batch_manager 구현

## 개요

대량의 파일 변환 및 처리를 효율적으로 관리하기 위한 모듈입니다. 작업을 배치(Batch) 단위로 나누어 실행하고, 각 단계의 진행 상황을 디스크(JSON 파일)에 영구 저장하여 상태 관리, 재시작, 모니터링을 지원합니다.

## 파일 경로

```
markitdown/app/batch_manager.py
```

## 주요 클래스

### 1. BatchStateManager

배치 작업의 전체 수명 주기(생성, 업데이트, 조회, 삭제)를 관리합니다.

- **상태 관리**: JSON 파일로 상태를 저장하여 서버 재시작 시에도 정보가 유지됩니다. (`load_state`, `save_state`)
- **주요 메서드**:
  - `create_batch(...)`: 파일 목록을 받아 배치 그룹으로 나눕니다. 고유한 Batch ID를 생성하고 초기 상태 파일을 만듭니다.
  - `update_batch_status(...)`: 배치 그룹(내부의 여러 파일 묶음) 전체의 상태(pending, processing, completed 등)를 업데이트합니다.
  - `update_file_status(...)`: 개별 파일의 처리 상태, 결과 경로, 인덱싱 여부 등을 기록합니다.
  - `get_next_pending_batch(...)`: 처리되지 않은 배치를 찾아 다음 작업을 스케줄링할 수 있게 합니다.
  - `_update_overall_status(...)`: 개별 배치의 진행 상황을 집계하여 전체 작업 진행률(%)과 최종 상태를 계산합니다.
  
## 동작 원리

1. **배치 생성**: 수백/수천 개의 파일 목록이 들어오면 `DEFAULT_BATCH_SIZE`(설정) 단위로 나눕니다.
2. **저장 구조**: `batch_state/` 디렉토리에 `{BATCH_ID}.json` 파일을 생성합니다.
3. **진행 추적**: 작업자는 파일을 하나씩 처리할 때마다 `update_file_status`를 호출하여 성공/실패를 기록합니다. 이 정보는 실시간으로 클라이언트가 조회(`/batch/{batch_id}`)할 수 있습니다.
4. **집계**: 모든 파일 처리가 끝나면 배치가 완료되고, 모든 배치가 끝나면 전체 작업이 완료(`completed`) 상태가 됩니다.

## 상태 스키마

- **BatchJobResponse**: 작업 전체 메타데이터
- **BatchInfo**: 개별 배치 그룹 정보
- **BatchFileStatus**: 파일별 상세 상태

## 데이터 흐름

1. 클라이언트 요청 (파일 목록) -> `create_batch` -> JSON 상태 파일 생성
2. 비동기 작업자 -> 파일 처리 -> `update_file_status` -> JSON 상태 업데이트
3. 클라이언트 조회 -> `get_batch_status` -> JSON 로드 -> 진행률 반환
