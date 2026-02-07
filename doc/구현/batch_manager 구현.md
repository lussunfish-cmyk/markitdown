# batch_manager.py 구현

## 개요

대량 파일 배치 처리를 위한 상태 관리 모듈입니다. 수백~수천 개의 파일을 배치로 나눠 처리하고, 각 파일의 처리 상태를 추적하며, 중단/재시작을 지원합니다.

## 파일 경로

```
markitdown/app/batch_manager.py
```

## 핵심 개념

### 배치 처리 Flow

```
파일 300개
    ↓
배치 분할 (100개씩)
    ↓
Batch 1 (100개) → 처리 → 상태 저장
Batch 2 (100개) → 처리 → 상태 저장
Batch 3 (100개) → 처리 → 상태 저장
    ↓
전체 완료
```

### 상태 저장 구조

```json
{
  "batch_id": "batch-20260207-143052-abc123",
  "total_files": 300,
  "total_batches": 3,
  "status": "processing",
  "progress_percentage": 65.3,
  "batches": [
    {
      "batch_num": 1,
      "status": "completed",
      "files": [
        {"filename": "file1.pdf", "status": "completed", "indexed": true},
        ...
      ]
    },
    ...
  ]
}
```

## 주요 구성 요소

### 1. BatchStateManager

배치 작업의 전체 lifecycle을 관리하는 핵심 클래스입니다.

```python
class BatchStateManager:
    """배치 작업 상태를 관리하는 클래스."""
    
    def __init__(self, state_dir: Optional[Path] = None):
        """
        상태 관리자를 초기화합니다.
        
        Args:
            state_dir: 배치 상태 파일 저장 경로 (None이면 기본값 사용)
        """
```

**주요 기능:**
- 배치 작업 생성 및 ID 할당
- 진행 상태를 JSON 파일로 저장/로드
- 파일별/배치별 상태 업데이트
- 전체 진행률 계산

---

### 2. 주요 메서드

#### 2.1 create_batch()

새로운 배치 작업을 생성합니다.

```python
def create_batch(
    self,
    files: List[str],
    batch_size: int,
    auto_index: bool = False
) -> str:
    """
    새 배치 작업 생성.
    
    Args:
        files: 처리할 파일 경로 목록
        batch_size: 배치당 파일 수 (예: 100)
        auto_index: 자동 인덱싱 여부
        
    Returns:
        batch_id (예: "batch-20260207-143052-abc123")
    """
```

**동작:**
1. 현재 시각 + 파일 해시로 고유 batch_id 생성
2. 파일 목록을 batch_size 단위로 분할
3. 각 배치 및 파일의 초기 상태를 "pending"으로 설정
4. JSON 파일로 저장

**예시:**
```python
manager = BatchStateManager()
batch_id = manager.create_batch(
    files=["file1.pdf", "file2.pdf", ..., "file300.pdf"],
    batch_size=100,
    auto_index=True
)
# 반환: "batch-20260207-143052-abc123"
```

---

#### 2.2 update_file_status()

개별 파일의 처리 상태를 업데이트합니다.

```python
def update_file_status(
    self,
    batch_id: str,
    batch_num: int,
    filename: str,
    status: str,
    **kwargs
) -> None:
    """
    개별 파일 처리 상태 업데이트.
    
    Args:
        batch_id: 배치 작업 ID
        batch_num: 배치 번호 (1부터 시작)
        filename: 파일명
        status: 새 상태 (pending, processing, completed, failed)
        **kwargs: 추가 필드 (error, converted_path, indexed, duration)
    """
```

**상태 종류:**
- `pending`: 처리 대기 중
- `processing`: 현재 처리 중
- `completed`: 처리 완료
- `failed`: 처리 실패

**예시:**
```python
# 성공 시
manager.update_file_status(
    batch_id="batch-20260207-143052-abc123",
    batch_num=1,
    filename="file1.pdf",
    status="completed",
    converted_path="output/file1.md",
    indexed=True,
    duration=5.2
)

# 실패 시
manager.update_file_status(
    batch_id="batch-20260207-143052-abc123",
    batch_num=1,
    filename="file2.pdf",
    status="failed",
    error="Invalid PDF format"
)
```

---

#### 2.3 update_batch_status()

배치 그룹의 상태를 업데이트합니다.

```python
def update_batch_status(
    self,
    batch_id: str,
    batch_num: int,
    status: str
) -> None:
    """
    배치 그룹의 상태를 업데이트.
    
    Args:
        batch_id: 배치 작업 ID
        batch_num: 배치 번호 (1부터 시작)
        status: 새 상태 (pending, processing, completed, failed)
    """
```

**사용 시점:**
- 배치 처리 시작 시: `status="processing"`
- 배치 완료 시: `status="completed"`

---

#### 2.4 load_state() / save_state()

배치 상태를 파일에서 읽고 쓰기합니다.

```python
def load_state(self, batch_id: str) -> Dict[str, Any]:
    """배치 상태를 로드."""
    
def save_state(self, batch_id: str, state: Dict[str, Any]) -> None:
    """배치 상태를 디스크에 저장."""
```

**저장 위치:**
```
/app/batch_state/batch-20260207-143052-abc123.json
```

---

#### 2.5 get_next_pending_batch()

다음 처리할 배치 번호를 반환합니다 (재시작용).

```python
def get_next_pending_batch(self, batch_id: str) -> Optional[int]:
    """
    처리 안된 첫 번째 배치 번호 반환.
    
    Returns:
        다음 pending 배치 번호 (없으면 None)
    """
```

**사용 예시:**
```python
# 중단된 배치 재시작
next_batch = manager.get_next_pending_batch(batch_id)
if next_batch:
    # next_batch부터 재처리
    pass
```

---

#### 2.6 delete_batch()

배치 작업을 삭제합니다.

```python
def delete_batch(self, batch_id: str) -> None:
    """배치 작업 삭제."""
```

---

#### 2.7 list_batches()

저장된 모든 배치 ID 목록을 반환합니다.

```python
def list_batches(self) -> List[str]:
    """저장된 모든 배치 ID 목록 반환."""
```

---

### 3. 내부 메서드

#### 3.1 _update_batch_stats()

배치의 통계 정보를 계산합니다.

```python
def _update_batch_stats(self, batch: Dict[str, Any]) -> None:
    """
    배치의 통계 정보 업데이트 (completed, failed 수).
    """
```

**계산 항목:**
- `completed`: 완료된 파일 수
- `failed`: 실패한 파일 수

---

#### 3.2 _update_overall_status()

전체 배치 작업의 상태와 진행률을 계산합니다.

```python
def _update_overall_status(self, state: Dict[str, Any]) -> None:
    """
    전체 배치 작업 상태 및 진행률 업데이트.
    """
```

**계산 로직:**
- 모든 배치가 `completed` → 전체 `completed`
- 하나라도 `processing` → 전체 `processing`
- 하나라도 `failed` → 전체 `failed`
- 진행률 = (완료 파일 수 / 전체 파일 수) × 100

---

### 4. 싱글톤 팩토리 함수

```python
def get_batch_manager() -> BatchStateManager:
    """싱글턴 배치 관리자 인스턴스를 반환합니다."""
```

전역적으로 하나의 BatchStateManager 인스턴스만 사용합니다.

---

## 사용 예시

### 1. 기본 배치 처리

```python
from app.batch_manager import get_batch_manager

# 배치 관리자 초기화
manager = get_batch_manager()

# 배치 작업 생성
files = ["file1.pdf", "file2.pdf", ..., "file300.pdf"]
batch_id = manager.create_batch(
    files=files,
    batch_size=100,
    auto_index=True
)

# 상태 로드
state = manager.load_state(batch_id)

# 각 배치 처리
for batch_info in state["batches"]:
    batch_num = batch_info["batch_num"]
    
    # 배치 시작
    manager.update_batch_status(batch_id, batch_num, "processing")
    
    # 각 파일 처리
    for file_info in batch_info["files"]:
        filename = file_info["filename"]
        
        try:
            # 파일 변환
            result = convert_file(filename)
            
            # 성공 시 상태 업데이트
            manager.update_file_status(
                batch_id, batch_num, filename,
                status="completed",
                converted_path=result["output"],
                duration=result["duration"]
            )
        except Exception as e:
            # 실패 시 상태 업데이트
            manager.update_file_status(
                batch_id, batch_num, filename,
                status="failed",
                error=str(e)
            )
    
    # 배치 완료
    manager.update_batch_status(batch_id, batch_num, "completed")

# 최종 상태 확인
final_state = manager.load_state(batch_id)
print(f"진행률: {final_state['progress_percentage']}%")
```

---

### 2. 중단된 배치 재시작

```python
# 기존 배치 ID로 재시작
batch_id = "batch-20260207-143052-abc123"

# 다음 처리할 배치 찾기
next_batch_num = manager.get_next_pending_batch(batch_id)

if next_batch_num:
    state = manager.load_state(batch_id)
    
    # next_batch_num부터 재처리
    for batch_info in state["batches"]:
        if batch_info["batch_num"] >= next_batch_num:
            # 처리 계속...
            pass
```

---

### 3. 배치 목록 조회

```python
# 모든 배치 목록
batch_ids = manager.list_batches()
print(f"저장된 배치: {len(batch_ids)}개")

for batch_id in batch_ids:
    state = manager.load_state(batch_id)
    print(f"{batch_id}: {state['status']} ({state['progress_percentage']}%)")
```

---

## 상태 파일 구조

### 저장 경로
```
/app/batch_state/
├── batch-20260207-143052-abc123.json
├── batch-20260207-150030-def456.json
└── batch-20260207-153015-ghi789.json
```

### JSON 구조
```json
{
  "batch_id": "batch-20260207-143052-abc123",
  "total_files": 300,
  "total_batches": 3,
  "batch_size": 100,
  "status": "processing",
  "auto_index": true,
  "started_at": "2026-02-07T14:30:52",
  "completed_at": null,
  "progress_percentage": 65.3,
  "batches": [
    {
      "batch_num": 1,
      "total_files": 100,
      "completed": 98,
      "failed": 2,
      "status": "completed",
      "files": [
        {
          "filename": "file1.pdf",
          "status": "completed",
          "error": null,
          "converted_path": "output/file1.md",
          "indexed": true,
          "duration": 5.2
        },
        {
          "filename": "file2.pdf",
          "status": "failed",
          "error": "Invalid PDF format",
          "converted_path": null,
          "indexed": false,
          "duration": 1.3
        }
      ]
    },
    {
      "batch_num": 2,
      "total_files": 100,
      "completed": 98,
      "failed": 0,
      "status": "processing",
      "files": [...]
    },
    {
      "batch_num": 3,
      "total_files": 100,
      "completed": 0,
      "failed": 0,
      "status": "pending",
      "files": [...]
    }
  ]
}
```

---

## FastAPI 통합

converter.py에서 BatchStateManager를 사용하는 엔드포인트:

### POST /convert-batch
```python
from .batch_manager import get_batch_manager

@app.post("/convert-batch")
async def convert_batch(
    files: List[UploadFile],
    batch_size: int = 100,
    auto_index: bool = True
) -> BatchJobResponse:
    manager = get_batch_manager()
    
    # 배치 생성
    batch_id = manager.create_batch(
        files=[f.filename for f in files],
        batch_size=batch_size,
        auto_index=auto_index
    )
    
    # 처리...
    
    final_state = manager.load_state(batch_id)
    return BatchJobResponse(**final_state)
```

### GET /batch/{batch_id}
```python
@app.get("/batch/{batch_id}")
async def get_batch_status(batch_id: str) -> BatchJobResponse:
    manager = get_batch_manager()
    
    if not manager.batch_exists(batch_id):
        raise HTTPException(404, "배치를 찾을 수 없습니다")
    
    state = manager.load_state(batch_id)
    return BatchJobResponse(**state)
```

### GET /batch
```python
@app.get("/batch")
async def list_batches() -> Dict[str, Any]:
    manager = get_batch_manager()
    batch_ids = manager.list_batches()
    
    batches = []
    for batch_id in batch_ids:
        state = manager.load_state(batch_id)
        batches.append({
            "batch_id": batch_id,
            "status": state["status"],
            "progress_percentage": state["progress_percentage"]
        })
    
    return {"total": len(batches), "batches": batches}
```

### DELETE /batch/{batch_id}
```python
@app.delete("/batch/{batch_id}")
async def delete_batch(batch_id: str):
    manager = get_batch_manager()
    manager.delete_batch(batch_id)
    return {"message": f"배치 {batch_id} 삭제 완료"}
```

---

## 설정

`config.py`의 BatchConfig에서 설정을 관리합니다:

```python
class BatchConfig:
    """배치 처리 설정."""
    
    # 배치 상태 저장 경로
    BATCH_STATE_DIR = Path(os.getenv("BATCH_STATE_DIR", "/app/batch_state"))
    
    # 기본 배치 크기
    DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "100"))
    
    # 배치 처리 타임아웃 (0이면 무제한)
    BATCH_TIMEOUT = int(os.getenv("BATCH_TIMEOUT", "0"))
```

### 환경 변수
```bash
BATCH_STATE_DIR=/app/batch_state
DEFAULT_BATCH_SIZE=100
BATCH_TIMEOUT=0
```

---

## 에러 처리

### FileNotFoundError
```python
try:
    state = manager.load_state("invalid-batch-id")
except FileNotFoundError:
    print("배치를 찾을 수 없습니다")
```

### 저장 실패
```python
try:
    manager.save_state(batch_id, state)
except Exception as e:
    logger.error(f"배치 상태 저장 실패: {e}")
```

---

## 장점

1. **체크포인트 기능**: 각 파일 처리마다 상태 저장 → 중단해도 재시작 가능
2. **진행률 추적**: 실시간으로 progress_percentage 계산
3. **에러 격리**: 한 파일 실패해도 다른 파일 계속 처리
4. **영구 저장**: JSON 파일로 저장 → 서버 재시작해도 상태 유지
5. **유연한 배치 크기**: batch_size 조절로 성능 최적화

---

## 제한 사항

1. **순차 처리**: 배치 내에서 병렬 처리 안 함 (동시성 문제 방지)
2. **디스크 I/O**: 파일별로 상태 저장 → 많은 디스크 쓰기
3. **메모리**: 모든 상태를 메모리에 로드 → 매우 큰 배치는 주의

---

## 성능 고려사항

### 권장 배치 크기
- **소규모 (10-100개)**: batch_size = 10-50
- **중규모 (100-1000개)**: batch_size = 100
- **대규모 (1000개 이상)**: batch_size = 100-200

### 디스크 사용량
- 파일 1개당 약 200-500 바이트 (JSON)
- 3000개 파일 = 약 0.6-1.5 MB

---

## 테스트

```python
# 소규모 테스트
manager = BatchStateManager()
batch_id = manager.create_batch(
    files=["test1.pdf", "test2.pdf", "test3.pdf"],
    batch_size=2,
    auto_index=False
)

state = manager.load_state(batch_id)
assert len(state["batches"]) == 2
assert state["total_files"] == 3
assert state["status"] == "pending"
```

---

## 참고

- **관련 모듈**: converter.py (API 엔드포인트)
- **스키마**: schemas.py (BatchJobResponse, BatchInfo, BatchFileStatus)
- **설정**: config.py (BatchConfig)
