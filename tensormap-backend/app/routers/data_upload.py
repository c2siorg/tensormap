import uuid as uuid_pkg
from pathlib import Path

from fastapi import APIRouter, Depends, Form, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from sqlmodel import Session, select
from werkzeug.utils import secure_filename

from app.config import get_settings
from app.database import get_db
from app.models import DataFile
from app.services.data_upload import (
    add_file_service,
    delete_one_file_by_id_service,
    get_all_files_service,
)
from app.shared.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["data-upload"])
ALLOWED_EXTENSIONS = {"csv", "zip"}


class _UploadFileWrapper:
    """Wraps FastAPI UploadFile to match the interface expected by service functions.

    The service reads the underlying stream via ``.file`` (Werkzeug-style) for its
    size check, so expose it directly rather than keeping a private ``._file``.
    """

    def __init__(self, upload_file: UploadFile):
        self.file = upload_file.file
        self.filename = upload_file.filename

    def save(self, path: str) -> None:
        self.file.seek(0)
        content = self.file.read()
        with open(path, "wb") as f:
            f.write(content)


@router.post("/data/upload/file")
def upload_file(
    data: UploadFile,
    project_id: uuid_pkg.UUID | None = Form(None),
    db: Session = Depends(get_db),
):
    """Upload a CSV file and persist its metadata."""
    logger.debug("Upload request: filename=%s, project_id=%s", data.filename, project_id)

    if not data.filename:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Please select the file.", "data": None},
        )

    ext = data.filename.rsplit(".", 1)[-1].lower() if "." in data.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Only CSV and ZIP files are supported.",
                "data": None,
            },
        )

    max_size = get_settings().max_content_length

    # Enforce the size limit without loading the entire file into memory.
    total_read = 0
    chunk_size = 1024 * 1024  # 1 MB

    while True:
        chunk = data.file.read(chunk_size)
        if not chunk:
            break
        total_read += len(chunk)
        if total_read > max_size:
            return JSONResponse(
                status_code=413,
                content={
                    "success": False,
                    "message": f"File too large. Maximum allowed size is {max_size // (1024 * 1024)} MB.",
                    "data": None,
                },
            )

    # Rewind the file so downstream code can re-read it from the beginning.
    data.file.seek(0)

    wrapper = _UploadFileWrapper(data)
    body, status_code = add_file_service(db, wrapper, project_id=project_id)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/data/upload/file")
def get_all_files(
    project_id: uuid_pkg.UUID | None = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """List uploaded files, optionally filtered by project."""
    body, status_code = get_all_files_service(db, project_id=project_id, offset=offset, limit=limit)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/data/upload/file/{file_id}/download")
def download_file(file_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
    """Download the current saved CSV contents for a dataset."""
    file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
    if file is None:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "Dataset was not found.", "data": None},
        )
    if file.file_type.lower() != "csv":
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Only CSV datasets can be downloaded as CSV.", "data": None},
        )

    upload_folder = Path(get_settings().upload_folder).resolve()
    file_path = (upload_folder / file.disk_name).resolve()
    if file_path == upload_folder or upload_folder not in file_path.parents or not file_path.is_file():
        logger.warning("CSV download file unavailable for dataset id=%s", file_id)
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "The dataset file is unavailable.", "data": None},
        )

    safe_name = secure_filename(Path(file.file_name).name)
    if safe_name.lower().endswith(".csv"):
        safe_name = safe_name[:-4]
    download_name = f"{safe_name or 'dataset'}.csv"
    return FileResponse(file_path, media_type="text/csv", filename=download_name)


@router.delete("/data/upload/file/{file_id}")
def delete_file(file_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
    """Delete an uploaded file by ID, removing both the DB record and the file on disk."""
    logger.debug("Deleting file with id: %s", file_id)
    body, status_code = delete_one_file_by_id_service(db, file_id=file_id)
    return JSONResponse(status_code=status_code, content=body)
