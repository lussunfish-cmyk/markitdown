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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI(title="MarkItDown API")

# Output directory for converted markdown files
OUTPUT_DIR = Path(os.getenv("MARKITDOWN_OUTPUT_DIR", "/app/output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input directory for batch processing
INPUT_DIR = Path(os.getenv("MARKITDOWN_INPUT_DIR", "/app/input"))
INPUT_DIR.mkdir(parents=True, exist_ok=True)

# Supported file extensions
SUPPORTED_FORMATS = {
    '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', 
    '.csv', '.json', '.xml', '.html', '.htm', '.txt', '.md',
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff',
    '.wav', '.mp3', '.m4a', '.flac', '.epub', '.zip'
}


class ConvertResponse(BaseModel):
    filename: str
    message: str


class FolderConvertResponse(BaseModel):
    total_files: int
    converted_files: int
    failed_files: int
    files: list[dict]
    message: str


def convert_doc_to_docx(doc_path: Path) -> tuple[Optional[Path], str]:
    """Convert .doc file to .docx using LibreOffice. Returns (path to .docx file, error message)."""
    try:
        # Use LibreOffice to convert .doc to .docx
        result = subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to", "docx",
                "--outdir", str(doc_path.parent),
                str(doc_path)
            ],
            capture_output=True,
            timeout=60,
            text=True
        )
        
        docx_path = doc_path.parent / f"{doc_path.stem}.docx"
        
        if result.returncode == 0 and docx_path.exists():
            return docx_path, ""
        else:
            error_output = result.stderr if result.stderr else result.stdout
            return None, f"LibreOffice conversion failed: {error_output}"
    except subprocess.TimeoutExpired:
        return None, "LibreOffice conversion timeout (>60s)"
    except FileNotFoundError:
        return None, "LibreOffice not found"
    except Exception as e:
        return None, f"Unexpected error during conversion: {str(e)}"


def convert_single_file(file_path: Path, log_progress: bool = False) -> tuple[bool, str, str, float]:
    """Convert a single file to markdown. Returns (success, filename, message, duration)."""
    start_time = time.time()
    
    if log_progress:
        logging.info(f"🔄 Converting: {file_path.name}")
    
    if file_path.suffix.lower() not in SUPPORTED_FORMATS:
        duration = time.time() - start_time
        if log_progress:
            logging.error(f"❌ Failed: {file_path.name} - Unsupported format ({duration:.2f}s)")
        return False, file_path.name, f"Unsupported format: {file_path.suffix}", duration

    actual_file_path = file_path
    temp_converted_docx = None

    try:
        # Convert .doc to .docx if needed
        if file_path.suffix.lower() == ".doc":
            if log_progress:
                logging.info(f"  📄 Converting .doc to .docx...")
            temp_converted_docx, error_msg = convert_doc_to_docx(file_path)
            if not temp_converted_docx:
                duration = time.time() - start_time
                if log_progress:
                    logging.error(f"❌ Failed: {file_path.name} - {error_msg} ({duration:.2f}s)")
                return False, file_path.name, f"Failed to convert .doc to .docx: {error_msg}", duration
            actual_file_path = temp_converted_docx

        converter = MarkItDown()
        result = converter.convert(str(actual_file_path))
        markdown_text = getattr(result, "text_content", None)
        if not markdown_text:
            markdown_text = getattr(result, "text", None)
        if not markdown_text:
            duration = time.time() - start_time
            if log_progress:
                logging.error(f"❌ Failed: {file_path.name} - No markdown extracted ({duration:.2f}s)")
            return False, file_path.name, "Failed to extract markdown", duration

        # Remove form feed characters
        markdown_text = markdown_text.replace('\f', '')

        output_filename = f"{file_path.stem}.md"
        output_path = OUTPUT_DIR / output_filename

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)

        duration = time.time() - start_time
        if log_progress:
            logging.info(f"✅ Success: {file_path.name} → {output_filename} ({duration:.2f}s)")
        return True, output_filename, "Converted successfully", duration
    except Exception as e:
        duration = time.time() - start_time
        if log_progress:
            logging.error(f"❌ Failed: {file_path.name} - {str(e)} ({duration:.2f}s)")
        return False, file_path.name, f"Error: {str(e)}", duration
    finally:
        # Clean up temporary .docx file if created
        if temp_converted_docx and temp_converted_docx.exists():
            try:
                temp_converted_docx.unlink()
            except Exception:
                pass


@app.post("/convert")
async def convert_file(file: UploadFile = File(...)) -> FileResponse:
    """Convert a single file and return it for download."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required")

    logging.info(f"📥 Received file: {file.filename}")
    
    input_suffix = Path(file.filename).suffix.lower() or ".bin"
    original_filename = Path(file.filename).stem

    with tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix) as tmp_in:
        tmp_in.write(await file.read())
        input_path = tmp_in.name

    try:
        success, output_filename, msg, duration = convert_single_file(Path(input_path), log_progress=True)
        if not success:
            raise HTTPException(status_code=500, detail=msg)
        
        # File is already saved in OUTPUT_DIR by convert_single_file
        output_path = OUTPUT_DIR / output_filename
        
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
    """Convert all supported files in the input directory and return results as JSON."""
    batch_start_time = time.time()
    
    if not INPUT_DIR.exists():
        raise HTTPException(status_code=400, detail="Input directory does not exist")

    files_to_convert = [
        f for f in INPUT_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    ]

    logging.info(f"📂 Starting batch conversion: {len(files_to_convert)} files found")
    logging.info("="*60)

    if not files_to_convert:
        result = {
            "total_files": 0,
            "converted_files": 0,
            "failed_files": 0,
            "total_duration": 0,
            "files": [],
            "message": "No supported files found in input directory"
        }
        
        # Save result to temporary JSON file
        result_path = OUTPUT_DIR / "conversion_result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return FileResponse(
            path=str(result_path),
            media_type="application/json",
            filename="conversion_result.json"
        )

    converted = []
    failed = []

    for idx, file_path in enumerate(sorted(files_to_convert), 1):
        logging.info(f"\n[{idx}/{len(files_to_convert)}]")
        success, output_filename, msg, duration = convert_single_file(file_path, log_progress=True)
        
        if success:
            converted.append({
                "input": file_path.name,
                "output": output_filename,
                "status": "success",
                "duration": round(duration, 2)
            })
        else:
            failed.append({
                "input": file_path.name,
                "status": "failed",
                "reason": msg,
                "duration": round(duration, 2)
            })

    total_duration = time.time() - batch_start_time
    
    logging.info("\n" + "="*60)
    logging.info(f"🏁 Batch conversion complete")
    logging.info(f"   Total: {len(files_to_convert)} files")
    logging.info(f"   ✅ Success: {len(converted)}")
    logging.info(f"   ❌ Failed: {len(failed)}")
    logging.info(f"   ⏱️  Total time: {total_duration:.2f}s")
    logging.info("="*60)

    all_results = converted + failed

    result = {
        "total_files": len(files_to_convert),
        "converted_files": len(converted),
        "failed_files": len(failed),
        "total_duration": round(total_duration, 2),
        "files": all_results,
        "message": f"Batch conversion complete: {len(converted)} succeeded, {len(failed)} failed"
    }
    
    # Save result to JSON file
    result_path = OUTPUT_DIR / "conversion_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logging.info(f"💾 Result saved to: {result_path}")
    
    return FileResponse(
        path=str(result_path),
        media_type="application/json",
        filename="conversion_result.json"
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats."""
    return {
        "formats": sorted(list(SUPPORTED_FORMATS)),
        "count": len(SUPPORTED_FORMATS)
    }
