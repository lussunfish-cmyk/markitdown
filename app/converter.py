"""
MarkItDown REST API Application
Converts various file formats to Markdown using the MarkItDown library.
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

# ============================================================================
# Configuration and Setup
# ============================================================================


class Config:
    """Application configuration."""
    OUTPUT_DIR = Path(os.getenv("MARKITDOWN_OUTPUT_DIR", "/app/output"))
    INPUT_DIR = Path(os.getenv("MARKITDOWN_INPUT_DIR", "/app/input"))
    
    SUPPORTED_FORMATS = {
        '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls',
        '.csv', '.json', '.xml', '.html', '.htm', '.txt', '.md',
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff',
        '.wav', '.mp3', '.m4a', '.flac', '.epub', '.zip'
    }
    
    LIBREOFFICE_TIMEOUT = 60
    RESULT_FILENAME = "conversion_result.json"
    
    # Ensure directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging() -> logging.Logger:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


logger = setup_logging()
app = FastAPI(title="MarkItDown API")

# ============================================================================
# Helper Functions
# ============================================================================


def is_supported_format(file_path: Path) -> bool:
    """Check if file format is supported."""
    return file_path.suffix.lower() in Config.SUPPORTED_FORMATS


def get_supported_files(directory: Path) -> list[Path]:
    """Get all supported files from directory recursively."""
    return [
        f for f in directory.rglob("*")
        if f.is_file() and is_supported_format(f)
    ]


def cleanup_temp_file(file_path: Optional[Path]) -> None:
    """Safely cleanup temporary file."""
    if file_path and file_path.exists():
        try:
            file_path.unlink()
        except Exception:
            pass

# ============================================================================
# Conversion Functions
# ============================================================================


def convert_doc_to_docx(doc_path: Path) -> tuple[Optional[Path], str]:
    """
    Convert .doc file to .docx using LibreOffice.
    
    Args:
        doc_path: Path to .doc file
        
    Returns:
        Tuple of (path to .docx file or None, error message or empty string)
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
            timeout=Config.LIBREOFFICE_TIMEOUT,
            text=True
        )
        
        docx_path = doc_path.parent / f"{doc_path.stem}.docx"
        
        if result.returncode == 0 and docx_path.exists():
            return docx_path, ""
        else:
            error_output = result.stderr if result.stderr else result.stdout
            return None, f"LibreOffice conversion failed: {error_output}"
    except subprocess.TimeoutExpired:
        return None, f"LibreOffice conversion timeout (>{Config.LIBREOFFICE_TIMEOUT}s)"
    except FileNotFoundError:
        return None, "LibreOffice not found"
    except Exception as e:
        return None, f"Unexpected error during conversion: {str(e)}"


def extract_markdown(file_path: Path) -> tuple[Optional[str], str]:
    """
    Extract markdown content from file.
    
    Args:
        file_path: Path to file to convert
        
    Returns:
        Tuple of (markdown text or None, error message or empty string)
    """
    try:
        converter = MarkItDown()
        result = converter.convert(str(file_path))
        
        markdown_text = getattr(result, "text_content", None)
        if not markdown_text:
            markdown_text = getattr(result, "text", None)
        
        if not markdown_text:
            return None, "Failed to extract markdown"
        
        # Clean form feed characters
        markdown_text = markdown_text.replace('\f', '')
        return markdown_text, ""
    except Exception as e:
        return None, f"Error extracting markdown: {str(e)}"


def save_markdown(markdown_text: str, output_filename: str) -> tuple[bool, str]:
    """
    Save markdown text to file.
    
    Args:
        markdown_text: Markdown content
        output_filename: Output filename
        
    Returns:
        Tuple of (success, error message or empty string)
    """
    try:
        output_path = Config.OUTPUT_DIR / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        return True, ""
    except Exception as e:
        return False, f"Error saving file: {str(e)}"


def convert_single_file(
    file_path: Path,
    log_progress: bool = False
) -> tuple[bool, str, str, float]:
    """
    Convert a single file to markdown.
    
    Args:
        file_path: Path to file to convert
        log_progress: Whether to log progress to console
        
    Returns:
        Tuple of (success, filename, message, duration in seconds)
    """
    start_time = time.time()
    
    if log_progress:
        logger.info(f"🔄 Converting: {file_path.name}")
    
    # Check if format is supported
    if not is_supported_format(file_path):
        duration = time.time() - start_time
        error_msg = f"Unsupported format: {file_path.suffix}"
        if log_progress:
            logger.error(f"❌ Failed: {file_path.name} - {error_msg} ({duration:.2f}s)")
        return False, file_path.name, error_msg, duration

    actual_file_path = file_path
    temp_converted_docx = None

    try:
        # Convert .doc to .docx if needed
        if file_path.suffix.lower() == ".doc":
            if log_progress:
                logger.info(f"  📄 Converting .doc to .docx...")
            
            temp_converted_docx, error_msg = convert_doc_to_docx(file_path)
            if not temp_converted_docx:
                duration = time.time() - start_time
                full_error = f"Failed to convert .doc to .docx: {error_msg}"
                if log_progress:
                    logger.error(f"❌ Failed: {file_path.name} - {error_msg} ({duration:.2f}s)")
                return False, file_path.name, full_error, duration
            
            actual_file_path = temp_converted_docx

        # Extract markdown
        markdown_text, extract_error = extract_markdown(actual_file_path)
        if not markdown_text:
            duration = time.time() - start_time
            if log_progress:
                logger.error(f"❌ Failed: {file_path.name} - {extract_error} ({duration:.2f}s)")
            return False, file_path.name, extract_error, duration

        # Save markdown
        output_filename = f"{file_path.stem}.md"
        success, save_error = save_markdown(markdown_text, output_filename)
        
        if not success:
            duration = time.time() - start_time
            if log_progress:
                logger.error(f"❌ Failed: {file_path.name} - {save_error} ({duration:.2f}s)")
            return False, file_path.name, save_error, duration

        duration = time.time() - start_time
        if log_progress:
            logger.info(f"✅ Success: {file_path.name} → {output_filename} ({duration:.2f}s)")
        
        return True, output_filename, "Converted successfully", duration

    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Unexpected error: {str(e)}"
        if log_progress:
            logger.error(f"❌ Failed: {file_path.name} - {error_msg} ({duration:.2f}s)")
        return False, file_path.name, error_msg, duration
    
    finally:
        cleanup_temp_file(temp_converted_docx)

# ============================================================================
# Result Handling
# ============================================================================


def save_result_json(result: dict) -> Path:
    """
    Save conversion result to JSON file.
    
    Args:
        result: Result dictionary
        
    Returns:
        Path to saved JSON file
    """
    result_path = Config.OUTPUT_DIR / Config.RESULT_FILENAME
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result_path


def log_batch_summary(total: int, converted: int, failed: int, duration: float) -> None:
    """Log batch conversion summary."""
    logger.info("\n" + "="*60)
    logger.info(f"🏁 Batch conversion complete")
    logger.info(f"   Total: {total} files")
    logger.info(f"   ✅ Success: {converted}")
    logger.info(f"   ❌ Failed: {failed}")
    logger.info(f"   ⏱️  Total time: {duration:.2f}s")
    logger.info("="*60)


def create_batch_result(
    total_files: int,
    converted_files: int,
    failed_files: int,
    total_duration: float,
    files: list[dict]
) -> dict:
    """Create batch conversion result dictionary."""
    return {
        "total_files": total_files,
        "converted_files": converted_files,
        "failed_files": failed_files,
        "total_duration": round(total_duration, 2),
        "files": files,
        "message": f"Batch conversion complete: {converted_files} succeeded, {failed_files} failed"
    }

# ============================================================================
# API Endpoints
# ============================================================================


@app.post("/convert")
async def convert_file(file: UploadFile = File(...)) -> FileResponse:
    """
    Convert a single uploaded file to markdown.
    
    The converted file is saved to the output directory and returned as a download.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required")

    logger.info(f"📥 Received file: {file.filename}")
    
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
        
        output_path = Config.OUTPUT_DIR / output_filename
        
        if not output_path.exists():
            raise HTTPException(status_code=500, detail="Converted file not found")
        
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
    Convert all supported files in the input directory.
    
    Returns a JSON file with conversion results and statistics.
    """
    batch_start_time = time.time()
    
    if not Config.INPUT_DIR.exists():
        raise HTTPException(status_code=400, detail="Input directory does not exist")

    files_to_convert = get_supported_files(Config.INPUT_DIR)

    logger.info(f"📂 Starting batch conversion: {len(files_to_convert)} files found")
    logger.info("="*60)

    # Handle empty directory
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
            filename=Config.RESULT_FILENAME
        )

    # Process files
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

    # Log summary
    log_batch_summary(
        len(files_to_convert),
        len(converted),
        len(failed),
        total_duration
    )

    # Create and save result
    result = create_batch_result(
        total_files=len(files_to_convert),
        converted_files=len(converted),
        failed_files=len(failed),
        total_duration=total_duration,
        files=all_results
    )
    
    result_path = save_result_json(result)
    logger.info(f"💾 Result saved to: {result_path}")
    
    return FileResponse(
        path=str(result_path),
        media_type="application/json",
        filename=Config.RESULT_FILENAME
    )


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/supported-formats")
async def get_supported_formats() -> dict:
    """Get list of supported file formats."""
    return {
        "formats": sorted(list(Config.SUPPORTED_FORMATS)),
        "count": len(Config.SUPPORTED_FORMATS)
    }
