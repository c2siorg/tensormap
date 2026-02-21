import os
import uuid as uuid_pkg
from typing import Any

import pandas as pd
from sqlalchemy import func
from sqlmodel import Session, select
from werkzeug.utils import secure_filename

from app.config import get_settings
from app.models import DataFile
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def _resp(status_code: int, success: bool, message: str, data: Any = None) -> tuple:
    """Build a standard API response tuple of (body_dict, status_code)."""
    return {"success": success, "message": message, "data": data}, status_code


def add_file_service(db: Session, file_wrapper: Any, project_id: uuid_pkg.UUID | None = None) -> tuple:
    """Save an uploaded file to disk and create a DataFile record."""
    settings = get_settings()
    upload_folder = settings.upload_folder
    os.makedirs(upload_folder, exist_ok=True)

    filename = secure_filename(file_wrapper.filename.lower())
    file_path = os.path.join(upload_folder, filename)
    file_wrapper.save(file_path)

    file_name_db = secure_filename(file_wrapper.filename.rsplit(".", 1)[0].lower())
    file_type_db = file_wrapper.filename.rsplit(".", 1)[1].lower()
    logger.info("Saving file: %s", file_name_db)
    record = DataFile(file_name=file_name_db, file_type=file_type_db, project_id=project_id)
    db.add(record)
    db.commit()
    return _resp(201, True, "File saved successfully")


def get_all_files_service(
    db: Session, project_id: uuid_pkg.UUID | None = None, offset: int = 0, limit: int = 50
) -> tuple:
    """Return a paginated list of uploaded files with their CSV column names."""
    settings = get_settings()
    upload_folder = settings.upload_folder

    try:
        base_filter = select(DataFile)
        if project_id is not None:
            base_filter = base_filter.where(DataFile.project_id == project_id)

        total = db.exec(select(func.count()).select_from(base_filter.subquery())).one()

        files = db.exec(base_filter.offset(offset).limit(limit)).all()
        data = []
        for file in files:
            try:
                df = pd.read_csv(f"{upload_folder}/{file.file_name}.{file.file_type}")
                fields = list(df.columns)
            except Exception:
                logger.warning("Failed to read CSV for file %s (id=%s)", file.file_name, file.id)
                fields = []
            data.append(
                {
                    "file_name": file.file_name,
                    "file_type": file.file_type,
                    "file_id": str(file.id),
                    "fields": fields,
                    "row_count": len(df) if fields else 0,
                    "error": fields == [],
                }
            )
        body = {"success": True, "message": "Saved files found successfully", "data": data}
        body["pagination"] = {"total": total, "offset": offset, "limit": limit}
        return body, 200
    except pd.errors.ParserError:
        logger.exception("CSV parsing error")
        return _resp(500, False, "An error occurred while parsing a CSV file")
    except Exception:
        logger.exception("Error fetching files")
        return _resp(500, False, "An error occurred while fetching the files")


def delete_one_file_by_id_service(db: Session, file_id: uuid_pkg.UUID) -> tuple:
    """Delete a file from disk and remove its DB record."""
    settings = get_settings()
    upload_folder = settings.upload_folder

    try:
        file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
        if not file:
            return _resp(400, False, "File not in the DB")

        file_path = os.path.join(upload_folder, f"{file.file_name}.{file.file_type}")

        if os.path.isfile(file_path):
            os.remove(file_path)
            db.delete(file)
            db.commit()
            return _resp(200, True, "File deleted successfully")
        else:
            return _resp(400, False, "File not found")
    except Exception:
        logger.exception("Error deleting file")
        return _resp(500, False, "An error occurred while deleting the file")
