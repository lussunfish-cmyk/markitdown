"""
파일 변환 REST API 애플리케이션.
다양한 파일 형식을 MarkItDown 라이브러리를 사용하여 마크다운으로 변환합니다.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from markitdown import MarkItDown
from .config import config
from .schemas import (
    ConversionFileResult,
    BatchConversionResult,
    SupportedFormatsResponse
)

# ============================================================================
# 설정 및 초기화
# ============================================================================


def setup_logging() -> logging.Logger:
    """애플리케이션 로깅을 설정합니다."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


logger = setup_logging()
app = FastAPI(title=config.API_TITLE)

# ============================================================================
# 보조 함수
# ============================================================================


def is_supported_format(file_path: Path) -> bool:
    """파일 형식이 지원되는지 확인합니다."""
    return file_path.suffix.lower() in config.CONVERSION.SUPPORTED_FORMATS


def get_supported_files(directory: Path) -> list[Path]:
    """디렉토리에서 지원되는 모든 파일을 재귀적으로 가져옵니다."""
    return [
        f for f in directory.rglob("*")
        if f.is_file() and is_supported_format(f)
    ]


def cleanup_temp_file(file_path: Optional[Path]) -> None:
    """임시 파일을 안전하게 정리합니다."""
    if file_path and file_path.exists():
        try:
            file_path.unlink()
        except Exception:
            pass

# ============================================================================
# 파일 변환 함수
# ============================================================================


def convert_doc_to_docx(doc_path: Path) -> tuple[Optional[Path], str]:
    """
    LibreOffice를 사용하여 .doc 파일을 .docx로 변환합니다.
    
    Args:
        doc_path: .doc 파일 경로
        
    Returns:
        (.docx 파일 경로 또는 None, 에러 메시지 또는 빈 문자열)의 튜플
    """
    try:
        result = subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to", "docx",
                "--outdir", str(doc_path.parent),
                str(doc_path)
            ],
            capture_output=True,
            timeout=config.CONVERSION.LIBREOFFICE_TIMEOUT,
            text=True
        )
        
        docx_path = doc_path.parent / f"{doc_path.stem}.docx"
        
        if result.returncode == 0 and docx_path.exists():
            return docx_path, ""
        else:
            error_output = result.stderr if result.stderr else result.stdout
            return None, f"LibreOffice 변환 실패: {error_output}"
    except subprocess.TimeoutExpired:
        return None, f"LibreOffice 변환 타임아웃 (>{config.CONVERSION.LIBREOFFICE_TIMEOUT}초)"
    except FileNotFoundError:
        return None, "LibreOffice를 찾을 수 없습니다"
    except Exception as e:
        return None, f"변환 중 예상치 못한 에러: {str(e)}"


def extract_markdown(file_path: Path) -> tuple[Optional[str], str]:
    """
    파일에서 마크다운 콘텐츠를 추출합니다.
    
    Args:
        file_path: 변환할 파일 경로
        
    Returns:
        (마크다운 텍스트 또는 None, 에러 메시지 또는 빈 문자열)의 튜플
    """
    try:
        converter = MarkItDown()
        result = converter.convert(str(file_path))
        
        markdown_text = getattr(result, "text_content", None)
        if not markdown_text:
            markdown_text = getattr(result, "text", None)
        
        if not markdown_text:
            return None, "마크다운 추출 실패"
        
        # 폼 피드 문자 제거
        markdown_text = markdown_text.replace('\f', '')
        return markdown_text, ""
    except Exception as e:
        return None, f"마크다운 추출 에러: {str(e)}"


def save_markdown(markdown_text: str, output_filename: str) -> tuple[bool, str]:
    """
    마크다운 텍스트를 파일로 저장합니다.
    
    Args:
        markdown_text: 마크다운 콘텐츠
        output_filename: 출력 파일명
        
    Returns:
        (성공 여부, 에러 메시지 또는 빈 문자열)의 튜플
    """
    try:
        output_path = config.CONVERSION.OUTPUT_DIR / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        return True, ""
    except Exception as e:
        return False, f"파일 저장 에러: {str(e)}"


def convert_single_file(
    file_path: Path,
    log_progress: bool = False
) -> tuple[bool, str, str, float]:
    """
    단일 파일을 마크다운으로 변환합니다.
    
    Args:
        file_path: 변환할 파일 경로
        log_progress: 진행 상황을 콘솔에 기록할지 여부
        
    Returns:
        (성공 여부, 파일명, 메시지, 소요 시간(초))의 튜플
    """
    start_time = time.time()
    
    if log_progress:
        logger.info(f"🔄 변환 중: {file_path.name}")
    
    # 파일 형식이 지원되는지 확인
    if not is_supported_format(file_path):
        duration = time.time() - start_time
        error_msg = f"지원하지 않는 형식: {file_path.suffix}"
        if log_progress:
            logger.error(f"❌ 실패: {file_path.name} - {error_msg} ({duration:.2f}초)")
        return False, file_path.name, error_msg, duration

    actual_file_path = file_path
    temp_converted_docx = None

    try:
        # .doc를 .docx로 변환 필요 시
        if file_path.suffix.lower() == ".doc":
            if log_progress:
                logger.info(f"  📄 .doc를 .docx로 변환 중...")
            
            temp_converted_docx, error_msg = convert_doc_to_docx(file_path)
            if not temp_converted_docx:
                duration = time.time() - start_time
                full_error = f".doc를 .docx로 변환 실패: {error_msg}"
                if log_progress:
                    logger.error(f"❌ 실패: {file_path.name} - {error_msg} ({duration:.2f}초)")
                return False, file_path.name, full_error, duration
            
            actual_file_path = temp_converted_docx

        # 마크다운 추출
        markdown_text, extract_error = extract_markdown(actual_file_path)
        if not markdown_text:
            duration = time.time() - start_time
            if log_progress:
                logger.error(f"❌ 실패: {file_path.name} - {extract_error} ({duration:.2f}초)")
            return False, file_path.name, extract_error, duration

        # 마크다운 저장
        output_filename = f"{file_path.stem}.md"
        success, save_error = save_markdown(markdown_text, output_filename)
        
        if not success:
            duration = time.time() - start_time
            if log_progress:
                logger.error(f"❌ 실패: {file_path.name} - {save_error} ({duration:.2f}초)")
            return False, file_path.name, save_error, duration

        duration = time.time() - start_time
        if log_progress:
            logger.info(f"✅ 성공: {file_path.name} → {output_filename} ({duration:.2f}초)")
        
        return True, output_filename, "성공적으로 변환됨", duration

    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"예상치 못한 에러: {str(e)}"
        if log_progress:
            logger.error(f"❌ 실패: {file_path.name} - {error_msg} ({duration:.2f}초)")
        return False, file_path.name, error_msg, duration
    
    finally:
        cleanup_temp_file(temp_converted_docx)

# ============================================================================
# 결과 처리
# ============================================================================


def save_result_json(result: dict) -> Path:
    """
    변환 결과를 JSON 파일로 저장합니다.
    
    Args:
        result: 결과 딕셔너리
        
    Returns:
        저장된 JSON 파일 경로
    """
    result_path = config.CONVERSION.OUTPUT_DIR / config.CONVERSION.RESULT_FILENAME
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result_path


def log_batch_summary(total: int, converted: int, failed: int, duration: float) -> None:
    """배치 변환 요약을 기록합니다."""
    logger.info("\n" + "="*60)
    logger.info(f"🏁 배치 변환 완료")
    logger.info(f"   총 파일: {total}개")
    logger.info(f"   ✅ 성공: {converted}개")
    logger.info(f"   ❌ 실패: {failed}개")
    logger.info(f"   ⏱️  총 소요 시간: {duration:.2f}초")
    logger.info("="*60)


def create_batch_result(
    total_files: int,
    converted_files: int,
    failed_files: int,
    total_duration: float,
    files: list[dict]
) -> dict:
    """배치 변환 결과 딕셔너리를 생성합니다."""
    return {
        "total_files": total_files,
        "converted_files": converted_files,
        "failed_files": failed_files,
        "total_duration": round(total_duration, 2),
        "files": files,
        "message": f"배치 변환 완료: {converted_files}개 성공, {failed_files}개 실패"
    }

# ============================================================================
# API 엔드포인트
# ============================================================================


@app.post("/convert")
async def convert_file(file: UploadFile = File(...)) -> FileResponse:
    """
    업로드된 단일 파일을 마크다운으로 변환합니다.
    
    변환된 파일은 출력 디렉토리에 저장되고 다운로드로 반환됩니다.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일명이 필요합니다")

    logger.info(f"📥 파일 수신됨: {file.filename}")
    
    input_suffix = Path(file.filename).suffix.lower() or ".bin"

    with tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix) as tmp_in:
        tmp_in.write(await file.read())
        input_path = tmp_in.name

    try:
        success, output_filename, msg, duration = convert_single_file(
            Path(input_path),
            log_progress=True
        )
        
        if not success:
            raise HTTPException(status_code=500, detail=msg)
        
        output_path = config.CONVERSION.OUTPUT_DIR / output_filename
        
        if not output_path.exists():
            raise HTTPException(status_code=500, detail="변환된 파일을 찾을 수 없습니다")
        
        return FileResponse(
            path=str(output_path),
            media_type="text/markdown",
            filename=output_filename
        )
    finally:
        try:
            os.remove(input_path)
        except FileNotFoundError:
            pass


@app.post("/convert-folder")
async def convert_folder() -> FileResponse:
    """
    입력 디렉토리의 모든 지원 파일을 변환합니다.
    
    변환 결과 및 통계가 포함된 JSON 파일을 반환합니다.
    """
    batch_start_time = time.time()
    
    if not config.CONVERSION.INPUT_DIR.exists():
        raise HTTPException(status_code=400, detail="입력 디렉토리가 존재하지 않습니다")

    files_to_convert = get_supported_files(config.CONVERSION.INPUT_DIR)

    logger.info(f"📂 배치 변환 시작: {len(files_to_convert)}개 파일 발견")
    logger.info("="*60)

    # 빈 디렉토리 처리
    if not files_to_convert:
        result = create_batch_result(
            total_files=0,
            converted_files=0,
            failed_files=0,
            total_duration=0,
            files=[]
        )
        result_path = save_result_json(result)
        
        return FileResponse(
            path=str(result_path),
            media_type="application/json",
            filename=config.CONVERSION.RESULT_FILENAME
        )

    # 파일 처리
    converted = []
    failed = []

    for idx, file_path in enumerate(sorted(files_to_convert), 1):
        logger.info(f"\n[{idx}/{len(files_to_convert)}]")
        success, output_filename, msg, duration = convert_single_file(
            file_path,
            log_progress=True
        )
        
        result_dict = {
            "input": file_path.name,
            "status": "success" if success else "failed",
            "duration": round(duration, 2)
        }
        
        if success:
            result_dict["output"] = output_filename
            converted.append(result_dict)
        else:
            result_dict["reason"] = msg
            failed.append(result_dict)

    total_duration = time.time() - batch_start_time
    all_results = converted + failed

    # 요약 기록
    log_batch_summary(
        len(files_to_convert),
        len(converted),
        len(failed),
        total_duration
    )

    # 결과 생성 및 저장
    result = create_batch_result(
        total_files=len(files_to_convert),
        converted_files=len(converted),
        failed_files=len(failed),
        total_duration=total_duration,
        files=all_results
    )
    
    result_path = save_result_json(result)
    logger.info(f"💾 결과 저장됨: {result_path}")
    
    return FileResponse(
        path=str(result_path),
        media_type="application/json",
        filename=config.CONVERSION.RESULT_FILENAME
    )


@app.get("/health")
async def health_check() -> dict:
    """헬스 체크 엔드포인트입니다."""
    return {"status": "healthy"}


@app.get("/supported-formats")
async def get_supported_formats() -> SupportedFormatsResponse:
    """지원하는 파일 형식 목록을 반환합니다."""
    return SupportedFormatsResponse(
        formats=sorted(list(config.CONVERSION.SUPPORTED_FORMATS)),
        count=len(config.CONVERSION.SUPPORTED_FORMATS)
    )
