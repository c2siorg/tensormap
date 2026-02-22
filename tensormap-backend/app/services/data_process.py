import uuid as uuid_pkg
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import func
from sqlmodel import Session, select

from app.config import get_settings
from app.models import DataFile, DataProcess
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def _resp(status_code: int, success: bool, message: str, data: Any = None) -> tuple:
    """Build a standard API response tuple of (body_dict, status_code)."""
    return {"success": success, "message": message, "data": data}, status_code


def _get_file_path(file: DataFile) -> str:
    """Resolve the on-disk path for a DataFile record."""
    settings = get_settings()
    return f"{settings.upload_folder}/{file.file_name}.{file.file_type}"


def add_target_service(db: Session, file_id: uuid_pkg.UUID, target: str) -> tuple:
    """Create a DataProcess record linking a file to its target column."""
    try:
        file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
        if not file:
            return _resp(400, False, "File doesn't exist in DB")

        data_process = DataProcess(file_id=file_id, target=target)
        db.add(data_process)
        db.commit()
        logger.info("Target field '%s' added for file_id=%s", target, file_id)
        return _resp(201, True, "Target field added successfully")
    except Exception as e:
        logger.exception("Error storing record: %s", str(e))
        return _resp(500, False, f"Error storing record: {e}")


def get_all_targets_service(db: Session, offset: int = 0, limit: int = 50) -> tuple:
    """Return all target field assignments with pagination."""
    total = db.exec(select(func.count()).select_from(DataProcess)).one()

    stmt = (
        select(DataProcess.file_id, DataFile.file_name, DataFile.file_type, DataProcess.target)
        .join(DataFile, DataFile.id == DataProcess.file_id)
        .offset(offset)
        .limit(limit)
    )
    rows = db.exec(stmt).all()
    data = [
        {
            "file_id": str(row.file_id),
            "file_name": row.file_name,
            "file_type": row.file_type,
            "target_field": row.target,
        }
        for row in rows
    ]
    body = {"success": True, "message": "Target fields of all files received successfully", "data": data}
    body["pagination"] = {"total": total, "offset": offset, "limit": limit}
    return body, 200


def delete_one_target_by_id_service(db: Session, file_id: uuid_pkg.UUID) -> tuple:
    """Delete the target field record for a given file."""
    file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
    if not file:
        return _resp(400, False, "File doesn't exist in DB")

    target_record = db.exec(select(DataProcess).where(DataProcess.file_id == file_id)).first()
    if not target_record:
        return _resp(400, False, "Target field doesn't exist")

    db.delete(target_record)
    db.commit()
    logger.info("Target field deleted for file_id=%s", file_id)
    return _resp(200, True, "Target field deleted successfully")


def get_one_target_by_id_service(db: Session, file_id: uuid_pkg.UUID) -> tuple:
    """Retrieve the target field for a single file."""
    file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
    if not file:
        return _resp(400, False, "File doesn't exist in DB")

    target_record = db.exec(select(DataProcess).where(DataProcess.file_id == file_id)).first()
    if not target_record:
        return _resp(400, False, "Target field doesn't exist")

    data = {
        "file_name": file.file_name,
        "file_type": file.file_type,
        "target_field": target_record.target,
    }
    return _resp(200, True, "Target fields of all files received successfully", data)


def get_data_metrics(db: Session, file_id: uuid_pkg.UUID) -> tuple:
    """Compute descriptive statistics and correlation matrix for a CSV dataset."""
    file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
    if not file:
        return _resp(400, False, "File doesn't exist in DB")

    file_path = _get_file_path(file)
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return _resp(500, False, f"File not found: {file_path}")
    except pd.errors.ParserError as e:
        logger.exception("CSV parsing error: %s", str(e))
        return _resp(500, False, f"Error reading CSV: {e}")
    except Exception as e:
        logger.exception("Error reading file: %s", str(e))
        return _resp(500, False, f"Error reading CSV: {e}")

    metrics = {
        "data_types": df.dtypes.apply(str).to_dict(),
        "correlation_matrix": df.corr(numeric_only=True).map(str).to_dict(),
        "metric": df.describe().map(str).to_dict(),
    }
    return _resp(200, True, "Dataset metrics generated successfully", metrics)


def get_file_data(db: Session, file_id: uuid_pkg.UUID) -> tuple:
    """Read and return the full contents of a CSV file as JSON."""
    file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
    if not file:
        return _resp(400, False, "Unable to open file")

    file_path = _get_file_path(file)
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return _resp(500, False, f"File not found: {file_path}")
    except pd.errors.ParserError as e:
        logger.exception("CSV parsing error: %s", str(e))
        return _resp(500, False, f"Error reading CSV: {e}")
    except Exception as e:
        logger.exception("Error reading file: %s", str(e))
        return _resp(500, False, f"Error reading CSV: {e}")

    data_json = df.to_json(orient="records")
    return _resp(200, True, "Data sent successfully", data_json)


_VALID_TRANSFORMATIONS = {"One Hot Encoding", "Categorical to Numerical", "Drop Column"}


def preprocess_data(db: Session, file_id: uuid_pkg.UUID, transformations: list) -> tuple:
    """Apply column transformations to a CSV, overwriting the existing file."""
    file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
    if not file:
        return _resp(400, False, "File doesn't exist in DB")

    file_path = _get_file_path(file)

    try:
        df = pd.read_csv(file_path)

        # Validate all transformations before applying any, so the request either
        # fully succeeds or fully fails — no partial mutations.
        for t in transformations:
            if t.transformation not in _VALID_TRANSFORMATIONS:
                return _resp(
                    422,
                    False,
                    f"Unknown transformation '{t.transformation}'. "
                    f"Valid options: {sorted(_VALID_TRANSFORMATIONS)}",
                )
            if t.feature not in df.columns:
                return _resp(
                    422,
                    False,
                    f"Column '{t.feature}' not found. "
                    f"Available columns: {list(df.columns)}",
                )

        for t in transformations:
            if t.transformation == "One Hot Encoding":
                df = pd.get_dummies(df, columns=[t.feature])
            if t.transformation == "Categorical to Numerical":
                df[t.feature] = pd.Categorical(df[t.feature]).codes
            if t.transformation == "Drop Column":
                df = df.drop(columns=[t.feature])
            if t.transformation == "Min-Max Normalization":
                col_min = df[t.feature].min()
                col_max = df[t.feature].max()
                df[t.feature] = 0.0 if col_max == col_min else (df[t.feature] - col_min) / (col_max - col_min)
            if t.transformation == "Z-score Standardization":
                std = df[t.feature].std()
                df[t.feature] = 0.0 if std == 0 else (df[t.feature] - df[t.feature].mean()) / std
            if t.transformation == "Log Transform":
                df[t.feature] = np.log1p(df[t.feature])
            if t.transformation == "Fill Missing Values":
                strategy = (t.params or {}).get("strategy", "mean")
                if strategy == "median":
                    df[t.feature] = df[t.feature].fillna(df[t.feature].median())
                elif strategy == "mode":
                    df[t.feature] = df[t.feature].fillna(df[t.feature].mode()[0])
                else:
                    df[t.feature] = df[t.feature].fillna(df[t.feature].mean())

        df.to_csv(file_path, index=False)
        return _resp(200, True, "Dataset preprocessed successfully")
    except pd.errors.ParserError as e:
        logger.exception("CSV parsing error: %s", str(e))
        return _resp(500, False, f"Error parsing CSV data: {e}")
    except Exception as e:
        logger.exception("Error preprocessing data: %s", str(e))
        return _resp(500, False, f"Error preprocessing data: {e}")
