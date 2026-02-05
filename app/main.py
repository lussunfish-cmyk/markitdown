from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from markitdown import MarkItDown

app = FastAPI(title="MarkItDown API")

# Output directory for converted markdown files
OUTPUT_DIR = Path(os.getenv("MARKITDOWN_OUTPUT_DIR", "/app/output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ConvertResponse(BaseModel):
    filename: str
    message: str


@app.post("/convert", response_model=ConvertResponse)
async def convert_file(file: UploadFile = File(...)) -> ConvertResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required")

    input_suffix = Path(file.filename).suffix or ".bin"

    with tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix) as tmp_in:
        tmp_in.write(await file.read())
        input_path = tmp_in.name

    try:
        converter = MarkItDown()
        result = converter.convert(input_path)
        markdown_text = getattr(result, "text_content", None)
        if not markdown_text:
            markdown_text = getattr(result, "text", None)
        if not markdown_text:
            raise HTTPException(status_code=500, detail="Failed to extract markdown")
    finally:
        try:
            os.remove(input_path)
        except FileNotFoundError:
            pass

    output_filename = f"{Path(file.filename).stem}.md"
    output_path = OUTPUT_DIR / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    return ConvertResponse(
        filename=output_filename,
        message=f"File converted successfully and saved to {output_path}",
    )
