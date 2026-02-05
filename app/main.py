from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from markitdown import MarkItDown

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


def convert_single_file(file_path: Path) -> tuple[bool, str, str]:
    """Convert a single file to markdown. Returns (success, filename, message)."""
    if file_path.suffix.lower() not in SUPPORTED_FORMATS:
        return False, file_path.name, f"Unsupported format: {file_path.suffix}"

    actual_file_path = file_path
    temp_converted_docx = None

    try:
        # Convert .doc to .docx if needed
        if file_path.suffix.lower() == ".doc":
            temp_converted_docx, error_msg = convert_doc_to_docx(file_path)
            if not temp_converted_docx:
                return False, file_path.name, f"Failed to convert .doc to .docx: {error_msg}"
            actual_file_path = temp_converted_docx

        converter = MarkItDown()
        result = converter.convert(str(actual_file_path))
        markdown_text = getattr(result, "text_content", None)
        if not markdown_text:
            markdown_text = getattr(result, "text", None)
        if not markdown_text:
            return False, file_path.name, "Failed to extract markdown"

        # Remove form feed characters
        markdown_text = markdown_text.replace('\f', '')

        output_filename = f"{file_path.stem}.md"
        output_path = OUTPUT_DIR / output_filename

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)

        return True, output_filename, "Converted successfully"
    except Exception as e:
        return False, file_path.name, f"Error: {str(e)}"
    finally:
        # Clean up temporary .docx file if created
        if temp_converted_docx and temp_converted_docx.exists():
            try:
                temp_converted_docx.unlink()
            except Exception:
                pass


@app.post("/convert", response_model=ConvertResponse)
async def convert_file(file: UploadFile = File(...)) -> ConvertResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required")

    input_suffix = Path(file.filename).suffix or ".bin"

    with tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix) as tmp_in:
        tmp_in.write(await file.read())
        input_path = tmp_in.name

    try:
        success, output_filename, msg = convert_single_file(Path(input_path))
        if not success:
            raise HTTPException(status_code=500, detail=msg)
    finally:
        try:
            os.remove(input_path)
        except FileNotFoundError:
            pass

    return ConvertResponse(
        filename=output_filename,
        message=msg,
    )


@app.post("/convert-folder", response_model=FolderConvertResponse)
async def convert_folder() -> FolderConvertResponse:
    """Convert all supported files in the input directory."""
    if not INPUT_DIR.exists():
        raise HTTPException(status_code=400, detail="Input directory does not exist")

    files_to_convert = [
        f for f in INPUT_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    ]

    if not files_to_convert:
        return FolderConvertResponse(
            total_files=0,
            converted_files=0,
            failed_files=0,
            files=[],
            message="No supported files found in input directory",
        )

    converted = []
    failed = []

    for file_path in sorted(files_to_convert):
        success, output_filename, msg = convert_single_file(file_path)
        if success:
            converted.append({
                "input": file_path.name,
                "output": output_filename,
                "status": "success"
            })
        else:
            failed.append({
                "input": file_path.name,
                "status": "failed",
                "reason": msg
            })

    all_results = converted + failed

    return FolderConvertResponse(
        total_files=len(files_to_convert),
        converted_files=len(converted),
        failed_files=len(failed),
        files=all_results,
        message=f"Batch conversion complete: {len(converted)} succeeded, {len(failed)} failed",
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
