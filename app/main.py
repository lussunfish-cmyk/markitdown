from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from markitdown import MarkItDown

app = FastAPI(title="MarkItDown API")


def _cleanup_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


@app.post("/convert")
async def convert_file(file: UploadFile = File(...)) -> FileResponse:
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
        _cleanup_file(input_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp_out:
        tmp_out.write(markdown_text.encode("utf-8"))
        output_path = tmp_out.name

    filename = f"{Path(file.filename).stem}.md"
    return FileResponse(
        output_path,
        media_type="text/markdown",
        filename=filename,
        background=BackgroundTask(_cleanup_file, output_path),
    )
