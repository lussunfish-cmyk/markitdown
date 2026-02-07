"""
배치 처리 상태 관리 모듈.
배치 작업의 진행 상황을 JSON 파일로 저장/로드.
"""

import json
import logging
import hashlib
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

from .config import config
from .schemas import BatchJobResponse, BatchInfo, BatchFileStatus

logger = logging.getLogger(__name__)


class BatchStateManager:
    """배치 작업 상태를 관리하는 클래스."""
    
    def __init__(self, state_dir: Optional[Path] = None):
        """
        상태 관리자를 초기화합니다.
        
        Args:
            state_dir: 배치 상태 파일 저장 경로 (None이면 기본값 사용)
        """
        self.state_dir = state_dir or config.BATCH.BATCH_STATE_DIR
        self.state_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"배치 상태 관리자 초기화: {self.state_dir}")
    
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
            batch_size: 배치당 파일 수
            auto_index: 자동 인덱싱 여부
            
        Returns:
            batch_id (예: "batch-20260207-143052-abc123")
        """
        # 배치 ID 생성
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        hash_suffix_length = config.BATCH.HASH_SUFFIX_LENGTH
        hash_suffix = hashlib.md5(str(files).encode()).hexdigest()[:hash_suffix_length]
        batch_id = f"batch-{timestamp}-{hash_suffix}"
        
        # 배치 분할
        total_batches = (len(files) + batch_size - 1) // batch_size
        batches = []
        
        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(files))
            batch_files = files[start_idx:end_idx]
            
            # 배치 정보 생성
            batch_info = BatchInfo(
                batch_num=i + 1,
                total_files=len(batch_files),
                status="pending",
                files=[
                    BatchFileStatus(filename=f, status="pending")
                    for f in batch_files
                ]
            )
            batches.append(batch_info)
        
        # 초기 상태 생성
        state = BatchJobResponse(
            batch_id=batch_id,
            total_files=len(files),
            total_batches=total_batches,
            batch_size=batch_size,
            status="pending",
            batches=batches,
            started_at=datetime.now().isoformat(),
            auto_index=auto_index
        )
        
        # 상태 저장
        self.save_state(batch_id, state.model_dump())
        logger.info(f"배치 생성 완료: {batch_id} ({len(files)}개 파일, {total_batches}개 배치)")
        
        return batch_id
    
    def save_state(self, batch_id: str, state: Dict[str, Any]) -> None:
        """
        배치 상태를 디스크에 저장.
        
        Args:
            batch_id: 배치 작업 ID
            state: 저장할 상태 딕셔너리
        """
        state_file = self.state_dir / f"{batch_id}.json"
        
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"배치 상태 저장 실패 ({batch_id}): {e}")
            raise
    
    def load_state(self, batch_id: str) -> Dict[str, Any]:
        """
        배치 상태를 로드.
        
        Args:
            batch_id: 배치 작업 ID
            
        Returns:
            상태 딕셔너리
        """
        state_file = self.state_dir / f"{batch_id}.json"
        
        if not state_file.exists():
            raise FileNotFoundError(f"배치를 찾을 수 없습니다: {batch_id}")
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"배치 상태 로드 실패 ({batch_id}): {e}")
            raise
    
    def batch_exists(self, batch_id: str) -> bool:
        """
        배치 작업이 존재하는지 확인.
        
        Args:
            batch_id: 배치 작업 ID
            
        Returns:
            존재 여부
        """
        state_file = self.state_dir / f"{batch_id}.json"
        return state_file.exists()
    
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
        state = self.load_state(batch_id)
        
        # 배치 찾기
        for batch in state["batches"]:
            if batch["batch_num"] == batch_num:
                batch["status"] = status
                break
        
        # 전체 상태 업데이트
        self._update_overall_status(state)
        
        # 저장
        self.save_state(batch_id, state)
        logger.debug(f"배치 {batch_num} 상태 업데이트: {status}")
    
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
            batch_num: 배치 번호
            filename: 파일명
            status: 새 상태
            **kwargs: 추가 필드 (error, converted_path, indexed, duration 등)
        """
        state = self.load_state(batch_id)
        
        # 배치 및 파일 찾기
        for batch in state["batches"]:
            if batch["batch_num"] == batch_num:
                for file_status in batch["files"]:
                    if file_status["filename"] == filename:
                        file_status["status"] = status
                        
                        # 추가 필드 업데이트
                        for key, value in kwargs.items():
                            file_status[key] = value
                        
                        # 배치 통계 업데이트
                        self._update_batch_stats(batch)
                        break
                break
        
        # 전체 상태 업데이트
        self._update_overall_status(state)
        
        # 저장
        self.save_state(batch_id, state)
    
    def get_next_pending_batch(self, batch_id: str) -> Optional[int]:
        """
        처리 안된 첫 번째 배치 번호 반환.
        
        Args:
            batch_id: 배치 작업 ID
            
        Returns:
            다음 pending 배치 번호 (없으면 None)
        """
        state = self.load_state(batch_id)
        
        for batch in state["batches"]:
            if batch["status"] in ["pending", "failed"]:
                return batch["batch_num"]
        
        return None
    
    def _update_batch_stats(self, batch: Dict[str, Any]) -> None:
        """
        배치의 통계 정보 업데이트 (completed, failed 수).
        
        Args:
            batch: 배치 딕셔너리
        """
        completed = sum(1 for f in batch["files"] if f["status"] == "completed")
        failed = sum(1 for f in batch["files"] if f["status"] == "failed")
        
        batch["completed"] = completed
        batch["failed"] = failed
    
    def _update_overall_status(self, state: Dict[str, Any]) -> None:
        """
        전체 배치 작업 상태 및 진행률 업데이트.
        
        Args:
            state: 상태 딕셔너리
        """
        # 모든 배치가 completed면 전체도 completed
        all_completed = all(b["status"] == "completed" for b in state["batches"])
        any_failed = any(b["status"] == "failed" for b in state["batches"])
        any_processing = any(b["status"] == "processing" for b in state["batches"])
        
        if all_completed:
            state["status"] = "completed"
            state["completed_at"] = datetime.now().isoformat()
        elif any_failed:
            state["status"] = "failed"
        elif any_processing:
            state["status"] = "processing"
        else:
            state["status"] = "pending"
        
        # 진행률 계산
        total_files = 0
        completed_files = 0
        
        for batch in state["batches"]:
            total_files += batch["total_files"]
            completed_files += batch["completed"]
        
        if total_files > 0:
            state["progress_percentage"] = round(
                (completed_files / total_files) * 100, 2
            )
        else:
            state["progress_percentage"] = 0.0
    
    def delete_batch(self, batch_id: str) -> None:
        """
        배치 작업 삭제.
        
        Args:
            batch_id: 배치 작업 ID
        """
        state_file = self.state_dir / f"{batch_id}.json"
        
        if state_file.exists():
            state_file.unlink()
            logger.info(f"배치 삭제 완료: {batch_id}")
        else:
            logger.warning(f"삭제할 배치를 찾을 수 없습니다: {batch_id}")
    
    def list_batches(self) -> List[str]:
        """
        저장된 모든 배치 ID 목록 반환.
        
        Returns:
            배치 ID 목록
        """
        batch_files = list(self.state_dir.glob("batch-*.json"))
        return [f.stem for f in batch_files]


# 전역 배치 관리자 인스턴스
_batch_manager: Optional[BatchStateManager] = None


def get_batch_manager() -> BatchStateManager:
    """싱글턴 배치 관리자 인스턴스를 반환합니다."""
    global _batch_manager
    if _batch_manager is None:
        _batch_manager = BatchStateManager()
    return _batch_manager
