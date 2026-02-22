import uuid as uuid_pkg

from fastapi import APIRouter, Depends, Form, Query, UploadFile, File
from fastapi.responses import JSONResponse
from sqlmodel import Session

from app.database import get_db
from app.services.data_upload import (
    add_file_service,
    delete_one_file_by_id_service,
    get_all_files_service,
)
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["data-upload"])

ALLOWED_EXTENSIONS = {"csv"}


class _UploadFileWrapper:
    """Wraps FastAPI UploadFile to match the interface expected by service functions."""

    def __init__(self, upload_file: UploadFile):
        self._file = upload_file
        self.filename = upload_file.filename

    def save(self, path: str) -> None:
        content = self._file.file.read()
        with open(path, "wb") as f:
            f.write(content)


@router.post("/data/upload/file")
async def upload_file(
    data: UploadFile = File(...),
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
                "message": "Only CSV files are supported.",
                "data": None,
            },
        )

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


@router.delete("/data/upload/file/{file_id}")
def delete_file(file_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
    """Delete an uploaded file by ID, removing both the DB record and the file on disk."""
    logger.debug("Deleting file with id: %s", file_id)
    body, status_code = delete_one_file_by_id_service(db, file_id=file_id)
    return JSONResponse(status_code=status_code, content=body)
