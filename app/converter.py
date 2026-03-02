"""
파일 변환 REST API 애플리케이션.
다양한 파일 형식을 MarkItDown 라이브러리를 사용하여 마크다운으로 변환합니다.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, HTTPException, UploadFile, Form, Query # type: ignore
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from pydantic import BaseModel  # type: ignore

from markitdown import MarkItDown   # type: ignore
from .config import config
from .schemas import (
    ConversionFileResult,
    BatchConversionResult,
    SupportedFormatsResponse,
    BatchJobResponse,
    IndexResponse,
    IndexFileRequest,
    IndexFolderRequest,
    RAGRequest,
    RAGResponse,
    RetrievalResult,
    SearchResult
)
from .schemas_eval import (
    TestsetGenerateRequest,
    TestsetGenerateResponse,
    EvaluationRequest,
    EvaluationResponse
)
from .evaluator import generate_testset_logic, evaluate_retrieval_logic

# ============================================================================
# 설정 및 초기화
# ============================================================================


def setup_logging() -> logging.Logger:
    """애플리케이션 로깅을 설정합니다."""
    logging.basicConfig(
        level=getattr(logging, config.LOGGING.LEVEL),
        format=config.LOGGING.FORMAT,
        datefmt=config.LOGGING.DATE_FORMAT
    )
    return logging.getLogger(__name__)


logger = setup_logging()
app = FastAPI(
    title=config.API.TITLE,
    version=config.API.VERSION,
    description=config.API.DESCRIPTION
)

# Static files 마운트 (프론트엔드)
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

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


def _docx_paragraph_text(paragraph: ET.Element, ns: dict[str, str]) -> str:
    texts: list[str] = []
    for node in paragraph.findall(".//w:t", ns):
        if node.text:
            texts.append(node.text)
    return "".join(texts).strip()


def extract_docx_markdown_fallback(file_path: Path) -> tuple[Optional[str], str]:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            try:
                xml_bytes = zf.read("word/document.xml")
            except KeyError:
                return None, "DOCX에 word/document.xml이 없습니다"

        root = ET.fromstring(xml_bytes)
        body = root.find("w:body", ns)
        if body is None:
            return None, "DOCX 본문을 찾을 수 없습니다"

        lines: list[str] = []
        for child in list(body):
            tag = child.tag.split("}")[-1]
            if tag == "p":
                paragraph_text = _docx_paragraph_text(child, ns)
                if paragraph_text:
                    lines.append(paragraph_text)
            elif tag == "tbl":
                for row in child.findall("w:tr", ns):
                    row_cells: list[str] = []
                    for cell in row.findall("w:tc", ns):
                        cell_parts: list[str] = []
                        for p in cell.findall("w:p", ns):
                            cell_text = _docx_paragraph_text(p, ns)
                            if cell_text:
                                cell_parts.append(cell_text)
                        row_cells.append(" ".join(cell_parts).strip())
                    if any(row_cells):
                        lines.append("| " + " | ".join(row_cells) + " |")

        markdown_text = "\n\n".join(line for line in lines if line)
        if not markdown_text.strip():
            return None, "DOCX fallback 추출 결과가 비어 있습니다"

        return markdown_text.replace('\f', ''), ""
    except Exception as e:
        return None, f"DOCX fallback 추출 에러: {str(e)}"

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
        # LibreOffice 경로 찾기: PATH에 있으면 사용, 없으면 macOS 기본 위치 확인
        libreoffice_cmd = shutil.which("libreoffice") or shutil.which("soffice")
        if not libreoffice_cmd and os.path.exists("/Applications/LibreOffice.app/Contents/MacOS/soffice"):
            libreoffice_cmd = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
        
        if not libreoffice_cmd:
            return None, "LibreOffice를 찾을 수 없습니다"
        
        result = subprocess.run(
            [
                libreoffice_cmd,
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
    except BaseException as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise

        if file_path.suffix.lower() == ".docx":
            fallback_text, fallback_error = extract_docx_markdown_fallback(file_path)
            if fallback_text:
                logger.warning(
                    f"⚠️ markitdown 변환 실패로 DOCX fallback 사용: {file_path.name} ({str(e)[:200]})"
                )
                return fallback_text, ""
            return None, f"마크다운 추출 에러: {str(e)} | fallback 실패: {fallback_error}"

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

    except BaseException as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise

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
async def convert_file(
    file: UploadFile = File(...),
    auto_index: bool = Form(False)
) -> Dict[str, Any]:
    """
    업로드된 단일 파일을 마크다운으로 변환합니다.
    
    Args:
        file: 변환할 파일
        auto_index: 변환 후 자동으로 인덱싱할지 여부
    
    Returns:
        변환 결과 및 인덱싱 정보
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일명이 필요합니다")

    logger.info(f"📥 파일 수신됨: {file.filename} (auto_index={auto_index})")
    
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
        
        result = {
            "filename": output_filename,
            "status": "converted",
            "output_path": str(output_path.relative_to(config.CONVERSION.OUTPUT_DIR)),
            "indexed": False,
            "duration": duration
        }
        
        # auto_index가 True면 즉시 인덱싱
        if auto_index:
            try:
                from .indexer import DocumentIndexer
                indexer = DocumentIndexer()
                index_result = indexer.index_file(output_path)
                
                result["status"] = "converted_and_indexed"
                result["indexed"] = True
                result["index_info"] = {
                    "document_id": output_path.stem,
                    "chunks": index_result.get("total_chunks", 0)
                }
                logger.info(f"✅ 인덱싱 완료: {output_filename}")
            except Exception as e:
                logger.error(f"❌ 인덱싱 실패: {e}")
                result["index_error"] = str(e)
        
        return result
        
    finally:
        try:
            os.remove(input_path)
        except FileNotFoundError:
            pass


@app.post("/convert-folder")
async def convert_folder(auto_index: bool = Form(False)) -> FileResponse:
    """
    입력 디렉토리의 모든 지원 파일을 변환합니다.
    
    Args:
        auto_index: 변환 후 자동으로 인덱싱할지 여부
    
    변환 결과 및 통계가 포함된 JSON 파일을 반환합니다.
    """
    batch_start_time = time.time()
    
    logger.info(f"📂 배치 변환 시작 (auto_index={auto_index})")
    
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
    converted_paths = []  # 인덱싱용

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
            converted_paths.append(config.CONVERSION.OUTPUT_DIR / output_filename)
        else:
            result_dict["reason"] = msg
            failed.append(result_dict)

    total_duration = time.time() - batch_start_time
    all_results = converted + failed

    # auto_index가 True면 변환된 파일들 인덱싱
    if auto_index and converted_paths:
        logger.info(f"\n✨ 자동 인덱싱 시작: {len(converted_paths)}개 파일")
        try:
            from .indexer import DocumentIndexer
            import time as time_module
            
            indexer = DocumentIndexer()
            
            indexed_count = 0
            failed_index_count = 0
            
            for i, md_path in enumerate(converted_paths, 1):
                try:
                    logger.info(f"[{i}/{len(converted_paths)}] 인덱싱 중: {md_path.name}")
                    result = indexer.index_file(md_path, force_reindex=False)
                    
                    if result.get("status") == "indexed":
                        indexed_count += 1
                        logger.info(f"✅ 인덱싱 완료: {md_path.name} ({result.get('chunks', 0)} 청크)")
                    elif result.get("status") == "skipped":
                        indexed_count += 1
                        logger.debug(f"⏭️  스킵: {md_path.name} (이미 인덱싱됨)")
                    else:
                        failed_index_count += 1
                        logger.warning(f"⚠️  인덱싱 실패: {md_path.name} - {result.get('message', '알 수 없는 에러')}")
                    
                    # ChromaDB readonly 에러 방지를 위해 짧은 딜레이
                    time_module.sleep(0.1)
                    
                except Exception as e:
                    failed_index_count += 1
                    logger.error(f"❌ 인덱싱 에러 ({md_path.name}): {e}")
            
            logger.info(f"\n✅ 인덱싱 완료: {indexed_count}개 성공, {failed_index_count}개 실패")
        except Exception as e:
            logger.error(f"인덱싱 초기화 에러: {e}")

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


@app.get("/", response_class=HTMLResponse)
async def root():
    """프론트엔드 UI를 제공합니다."""
    static_path = Path(__file__).parent / "static" / "index.html"
    if static_path.exists():
        return HTMLResponse(content=static_path.read_text(), status_code=200)
    return HTMLResponse(content="<h1>MarkItDown API</h1><p>Frontend not found</p>", status_code=200)


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


# ============================================================================
# 배치 처리 엔드포인트
# ============================================================================

@app.post("/convert-batch")
async def convert_batch(
    files: List[UploadFile] = File(...),
    batch_size: int = Form(100),
    auto_index: bool = Form(True)
) -> BatchJobResponse:
    """
    여러 파일을 배치로 나눠서 처리.
    
    Args:
        files: 업로드된 파일 목록
        batch_size: 배치당 처리할 파일 수
        auto_index: 변환 후 자동 임베딩 여부
    
    Returns:
        배치 작업 정보 및 상태
    """
    from .batch_manager import get_batch_manager
    from .indexer import DocumentIndexer
    
    logger.info(f"📦 배치 작업 시작: {len(files)}개 파일, batch_size={batch_size}, auto_index={auto_index}")
    
    # 파일들을 임시 저장
    temp_files = []
    for file in files:
        if not file.filename:
            continue
        
        input_suffix = Path(file.filename).suffix.lower() or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix, dir="/tmp") as tmp:
            tmp.write(await file.read())
            temp_files.append((Path(tmp.name), file.filename))
    
    # 배치 작업 생성
    batch_manager = get_batch_manager()
    batch_id = batch_manager.create_batch(
        files=[name for _, name in temp_files],
        batch_size=batch_size,
        auto_index=auto_index
    )
    
    # 배치별로 순차 처리
    state = batch_manager.load_state(batch_id)
    indexer = DocumentIndexer() if auto_index else None
    
    for batch_info in state["batches"]:
        batch_num = batch_info["batch_num"]
        
        # 배치 시작
        batch_manager.update_batch_status(batch_id, batch_num, "processing")
        
        for file_info in batch_info["files"]:
            filename = file_info["filename"]
            
            # 해당 파일의 임시 경로 찾기
            temp_path = None
            for tmp_path, orig_name in temp_files:
                if orig_name == filename:
                    temp_path = tmp_path
                    break
            
            if not temp_path:
                batch_manager.update_file_status(
                    batch_id, batch_num, filename,
                    status="failed",
                    error="임시 파일을 찾을 수 없습니다"
                )
                continue
            
            try:
                # 변환
                start_time = time.time()
                success, output_filename, msg, duration = convert_single_file(
                    temp_path,
                    log_progress=False
                )
                
                if not success:
                    batch_manager.update_file_status(
                        batch_id, batch_num, filename,
                        status="failed",
                        error=msg,
                        duration=duration
                    )
                    continue
                
                md_path = config.CONVERSION.OUTPUT_DIR / output_filename
                
                # 임베딩 (옵션)
                indexed = False
                if auto_index and indexer:
                    try:
                        indexer.index_file(md_path)
                        indexed = True
                    except Exception as e:
                        logger.error(f"인덱싱 실패 ({filename}): {e}")
                
                # 성공 상태 업데이트
                batch_manager.update_file_status(
                    batch_id, batch_num, filename,
                    status="completed",
                    converted_path=output_filename,
                    indexed=indexed,
                    duration=duration
                )
                
            except Exception as e:
                # 실패 상태 업데이트
                batch_manager.update_file_status(
                    batch_id, batch_num, filename,
                    status="failed",
                    error=str(e)
                )
        
        # 배치 완료
        batch_manager.update_batch_status(batch_id, batch_num, "completed")
    
    # 임시 파일 정리
    for tmp_path, _ in temp_files:
        try:
            tmp_path.unlink()
        except:
            pass
    
    # 최종 상태 반환
    final_state = batch_manager.load_state(batch_id)
    return BatchJobResponse(**final_state)


@app.get("/batch/{batch_id}")
async def get_batch_status(batch_id: str) -> BatchJobResponse:
    """
    배치 작업 상태 조회.
    
    Args:
        batch_id: 배치 작업 ID
    
    Returns:
        현재 배치 작업 상태
    """
    from .batch_manager import get_batch_manager
    
    batch_manager = get_batch_manager()
    
    if not batch_manager.batch_exists(batch_id):
        raise HTTPException(404, "배치 작업을 찾을 수 없습니다")
    
    state = batch_manager.load_state(batch_id)
    return BatchJobResponse(**state)


@app.delete("/batch/{batch_id}")
async def delete_batch(batch_id: str) -> Dict[str, str]:
    """
    배치 작업 삭제.
    
    Args:
        batch_id: 배치 작업 ID
    
    Returns:
        삭제 확인 메시지
    """
    from .batch_manager import get_batch_manager
    
    batch_manager = get_batch_manager()
    
    if not batch_manager.batch_exists(batch_id):
        raise HTTPException(404, "배치 작업을 찾을 수 없습니다")
    
    batch_manager.delete_batch(batch_id)
    return {"message": f"배치 {batch_id} 삭제 완료"}


@app.post("/batch/folder")
async def batch_folder(
    batch_size: int = Query(100, description="배치당 파일 수"),
    auto_index: bool = Query(True, description="자동 인덱싱 여부")
) -> BatchJobResponse:
    """
    input 폴더의 모든 파일을 배치 처리 (파일 업로드 없이).
    
    Args:
        batch_size: 배치당 처리할 파일 수
        auto_index: 변환 후 자동 임베딩 여부
    
    Returns:
        배치 작업 정보 및 상태
    """
    from .batch_manager import get_batch_manager
    from .indexer import DocumentIndexer
    
    if not config.CONVERSION.INPUT_DIR.exists():
        raise HTTPException(400, "입력 디렉토리가 존재하지 않습니다")
    
    files_to_convert = get_supported_files(config.CONVERSION.INPUT_DIR)
    
    if not files_to_convert:
        raise HTTPException(400, "변환할 파일이 없습니다")
    
    logger.info(f"📦 Input 폴더 배치 작업 시작: {len(files_to_convert)}개 파일, batch_size={batch_size}, auto_index={auto_index}")
    
    # 배치 작업 생성
    batch_manager = get_batch_manager()
    batch_id = batch_manager.create_batch(
        files=[str(f) for f in files_to_convert],
        batch_size=batch_size,
        auto_index=auto_index
    )
    
    # 배치별로 순차 처리
    state = batch_manager.load_state(batch_id)
    indexer = DocumentIndexer() if auto_index else None
    
    for batch_info in state["batches"]:
        batch_num = batch_info["batch_num"]
        
        # 배치 시작
        batch_manager.update_batch_status(batch_id, batch_num, "processing")
        
        for file_info in batch_info["files"]:
            filename = file_info["filename"]
            
            # 파일 경로 찾기
            file_path = Path(filename)
            if not file_path.exists():
                # 상대 경로인 경우 input 디렉토리 기준으로 찾기
                file_path = config.CONVERSION.INPUT_DIR / Path(filename).name
            
            if not file_path.exists():
                batch_manager.update_file_status(
                    batch_id, batch_num, filename,
                    status="failed",
                    error="파일을 찾을 수 없습니다"
                )
                continue
            
            try:
                # 변환
                success, output_filename, msg, duration = convert_single_file(
                    file_path,
                    log_progress=True
                )
                
                if not success:
                    batch_manager.update_file_status(
                        batch_id, batch_num, filename,
                        status="failed",
                        error=msg,
                        duration=duration
                    )
                    continue
                
                md_path = config.CONVERSION.OUTPUT_DIR / output_filename
                
                # 임베딩 (옵션)
                indexed = False
                if auto_index and indexer:
                    try:
                        indexer.index_file(md_path)
                        indexed = True
                    except Exception as e:
                        logger.error(f"인덱싱 실패 ({filename}): {e}")
                
                # 성공 상태 업데이트
                batch_manager.update_file_status(
                    batch_id, batch_num, filename,
                    status="completed",
                    converted_path=output_filename,
                    indexed=indexed,
                    duration=duration
                )
                
            except Exception as e:
                # 실패 상태 업데이트
                batch_manager.update_file_status(
                    batch_id, batch_num, filename,
                    status="failed",
                    error=str(e)
                )
        
        # 배치 완료
        batch_manager.update_batch_status(batch_id, batch_num, "completed")
    
    # 최종 상태 반환
    final_state = batch_manager.load_state(batch_id)
    return BatchJobResponse(**final_state)


@app.get("/batch")
async def list_batches() -> Dict[str, Any]:
    """
    저장된 모든 배치 작업 목록 조회.
    
    Returns:
        배치 ID 목록 및 요약 정보
    """
    from .batch_manager import get_batch_manager
    
    batch_manager = get_batch_manager()
    batch_ids = batch_manager.list_batches()
    
    batches_summary = []
    for batch_id in batch_ids:
        try:
            state = batch_manager.load_state(batch_id)
            batches_summary.append({
                "batch_id": batch_id,
                "status": state.get("status"),
                "total_files": state.get("total_files"),
                "progress_percentage": state.get("progress_percentage"),
                "started_at": state.get("started_at")
            })
        except Exception as e:
            logger.error(f"배치 {batch_id} 로드 실패: {e}")
    
    return {
        "total_batches": len(batches_summary),
        "batches": batches_summary
    }


# ============================================================================
# 인덱싱 엔드포인트
# ============================================================================

@app.post("/index")
async def index_document(request: IndexFileRequest) -> Dict[str, Any]:
    """
    단일 마크다운 파일 인덱싱.
    
    Args:
        request: 인덱싱 요청 (file_path, force)
    
    Returns:
        인덱싱 결과
    """
    from .indexer import DocumentIndexer
    
    indexer = DocumentIndexer()
    full_path = config.CONVERSION.OUTPUT_DIR / request.file_path
    
    if not full_path.exists():
        raise HTTPException(404, f"파일을 찾을 수 없습니다: {request.file_path}")
    
    try:
        result = indexer.index_file(full_path, force_reindex=request.force)
        return {
            "status": "success",
            "message": f"인덱싱 완료: {request.file_path}",
            "result": result
        }
    except Exception as e:
        logger.error(f"인덱싱 실패: {e}")
        raise HTTPException(500, f"인덱싱 실패: {str(e)}")


@app.post("/index-folder")
async def index_folder(request: IndexFolderRequest) -> IndexResponse:
    """
    output 폴더의 모든 MD 파일 인덱싱.
    
    Args:
        request: 폴더 인덱싱 요청 (folder, force)
    
    Returns:
        인덱싱 결과
    """
    from .indexer import DocumentIndexer
    
    indexer = DocumentIndexer()
    target_dir = config.CONVERSION.OUTPUT_DIR / request.folder
    
    if not target_dir.exists():
        raise HTTPException(404, f"폴더를 찾을 수 없습니다: {request.folder}")
    
    try:
        result = indexer.index_directory(target_dir, force_reindex=request.force)
        return IndexResponse(**result.model_dump()) if hasattr(result, 'model_dump') else IndexResponse(**result)
    except Exception as e:
        logger.error(f"폴더 인덱싱 실패: {e}")
        raise HTTPException(500, f"폴더 인덱싱 실패: {str(e)}")


@app.get("/documents")
async def list_documents() -> Dict[str, Any]:
    """
    인덱싱된 문서 목록 조회.
    
    Returns:
        문서 목록 및 통계
    """
    from .indexer import DocumentIndexer
    
    try:
        indexer = DocumentIndexer()
        docs = indexer.get_indexed_files()
        
        return {
            "total": len(docs),
            "documents": [d.model_dump() for d in docs]
        }
    except Exception as e:
        logger.error(f"문서 목록 조회 실패: {e}")
        raise HTTPException(500, f"문서 목록 조회 실패: {str(e)}")


# ============================================================================
# RAG 엔드포인트
# ============================================================================

@app.post("/query")
async def query(request: RAGRequest) -> RAGResponse:
    """
    RAG 질의응답.
    
    Args:
        request: 질의 요청 (query, top_k 등)
    
    Returns:
        답변 및 출처
    """
    from .rag import RAGPipeline
    
    try:
        rag = RAGPipeline()
        rag_result = rag.query(
            question=request.query,
            top_k=request.top_k,
            include_sources=request.include_sources
        )
        
        # RAGResult를 RAGResponse로 변환
        response = RAGResponse(
            answer=rag_result.answer,
            sources=[
                RetrievalResult(
                    content=src.get("content", ""),
                    source=src.get("source", ""),
                    chunk_id=src.get("chunk_id", 0),
                    similarity_score=src.get("score", 0.0)
                )
                for src in rag_result.sources
            ],
            model=config.OLLAMA.LLM_MODEL,
            tokens_used=None
        )
        
        # 디버깅: 첫 번째 출처 내용 확인
        if response.sources:
            logger.info(f"첫 번째 출처 - content 길이: {len(response.sources[0].content)}, source: {response.sources[0].source}")
        
        return response
    except Exception as e:
        logger.error(f"RAG 질의 실패: {e}")
        raise HTTPException(500, f"RAG 질의 실패: {str(e)}")


@app.get("/search")
async def search_documents(
    query: str = Query(..., description="검색 쿼리"),
    top_k: int = Query(5, description="반환할 문서 수")
) -> Dict[str, Any]:
    """
    문서 검색만 (답변 생성 없음).
    
    Args:
        query: 검색 쿼리
        top_k: 반환할 문서 수
    
    Returns:
        검색된 문서 목록
    """
    from .retriever import get_retriever
    
    try:
        retriever = get_retriever()
        results = retriever.search(query, k=top_k)
        
        return {
            "query": query,
            "total_results": len(results),
            "results": [
                {
                    "id": r.id,
                    "content": r.content,
                    "score": r.score,
                    "metadata": r.metadata
                }
                for r in results
            ]
        }
    except Exception as e:
        logger.error(f"문서 검색 실패: {e}")
        raise HTTPException(500, f"문서 검색 실패: {str(e)}")


@app.get("/testset/download")
async def download_testset(file_path: str = Query(..., description="다운로드할 테스트셋 파일 경로")):
    """
    생성된 테스트셋 CSV 파일을 다운로드합니다.
    """
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),
        media_type='text/csv'
    )


@app.post("/testset/generate", response_model=TestsetGenerateResponse)
async def generate_testset(request: TestsetGenerateRequest):
    """
    테스트셋 생성 API
    
    지정된 입력 디렉토리의 문서들을 바탕으로 Ragas를 사용하여 테스트 데이터셋(질문-답변 쌍)을 생성합니다.
    """
    try:
        sample_preview = await generate_testset_logic(
            input_dir=request.input_dir,
            output_file=request.output_file,
            test_size=request.test_size
        )
        return TestsetGenerateResponse(
            status="success",
            message=f"Testset generated (saved to {request.output_file})",
            output_file=request.output_file,
            sample_preview=sample_preview
        )
    except Exception as e:
        logger.error(f"Testset generation failed: {e}")
        raise HTTPException(500, f"Testset generation failed: {str(e)}")


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_retrieval(
    file: UploadFile = File(..., description="수동으로 검증/수정된 테스트셋 CSV 파일"),
    top_k: int = Form(5, description="검색할 상위 문서 수")
):
    """
    검색 성능 평가 API
    
    업로드된 테스트셋(CSV)을 사용하여 RAG 검색 시스템의 성능(Recall, Precision 등)을 평가합니다.
    (사용자가 편집한 testset.csv를 업로드하여 평가 수행)
    """
    try:
        # 1. 파일 저장 (output 디렉토리에 저장하여 결과 파일도 영속시킴)
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"eval_upload_{timestamp}_{file.filename}"
        saved_file_path = output_dir / safe_filename
        
        with open(saved_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"Evaluation file saved to: {saved_file_path}")

        # 2. 평가 로직 실행
        metrics = await evaluate_retrieval_logic(
            testset_path=str(saved_file_path),
            top_k=top_k
        )
        
        results_file = metrics.get("detail_file", "")
        
        return EvaluationResponse(
            status="success",
            message="Evaluation completed",
            results_file=results_file,
            metrics=metrics
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(500, f"Evaluation failed: {str(e)}")

