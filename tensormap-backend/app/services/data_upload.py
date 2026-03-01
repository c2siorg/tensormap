import os
import uuid as uuid_pkg
from typing import Any

import pandas as pd
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
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

    # Cache column names and row count at upload time (CSV only)
    columns_list: list[str] | None = None
    row_count: int | None = None
    if file_type_db == "csv":
        try:
            df_header = pd.read_csv(file_path, nrows=0)
            columns_list = list(df_header.columns)
            row_count = sum(chunk.shape[0] for chunk in pd.read_csv(file_path, chunksize=10_000))
        except (pd.errors.ParserError, OSError):
            logger.warning("Could not extract columns/row_count from %s", file_path)

    record = DataFile(
        file_name=file_name_db,
        file_type=file_type_db,
        project_id=project_id,
        columns=columns_list,
        row_count=row_count,
    )
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
            # Prefer cached columns/row_count from the DB
            if file.columns is not None:
                fields = file.columns
                row_count = file.row_count or 0
            elif file.file_type == "csv":
                # Backfill: read header + count rows, then persist the cache
                try:
                    file_path = f"{upload_folder}/{file.file_name}.{file.file_type}"
                    df_header = pd.read_csv(file_path, nrows=0)
                    fields = list(df_header.columns)
                    row_count = sum(chunk.shape[0] for chunk in pd.read_csv(file_path, chunksize=10_000))
                    file.columns = fields
                    file.row_count = row_count
                    db.add(file)
                    db.commit()
                except (pd.errors.ParserError, OSError):
                    logger.warning("Failed to read CSV for file %s (id=%s)", file.file_name, file.id)
                    fields = []
                    row_count = 0
            else:
                # Non-CSV files (e.g. ZIP) have no columns to cache
                fields = []
                row_count = 0
            data.append(
                {
                    "file_name": file.file_name,
                    "file_type": file.file_type,
                    "file_id": str(file.id),
                    "fields": fields,
                    "row_count": row_count,
                    "error": fields == [],
                }
            )
        body = {"success": True, "message": "Saved files found successfully", "data": data}
        body["pagination"] = {"total": total, "offset": offset, "limit": limit}
        return body, 200
    except pd.errors.ParserError:
        logger.exception("CSV parsing error")
        return _resp(500, False, "An error occurred while parsing a CSV file")
    except SQLAlchemyError:
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
    except (SQLAlchemyError, OSError):
        logger.exception("Error deleting file")
        return _resp(500, False, "An error occurred while deleting the file")
