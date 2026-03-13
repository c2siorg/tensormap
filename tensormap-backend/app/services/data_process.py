import uuid as uuid_pkg
from collections.abc import Callable
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


def get_column_stats_service(db: Session, file_id: uuid_pkg.UUID) -> tuple:
    """Compute per-column descriptive statistics for a CSV dataset."""
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

    total_rows = len(df)
    total_cols = len(df.columns)

    def _safe_float(v) -> float | None:
        """Return float(v) if v is a finite number, else None."""
        return float(v) if pd.notna(v) else None

    numeric_cols = df.select_dtypes(include="number").columns
    columns = []
    for col in df.columns:
        is_numeric = col in numeric_cols
        null_count = int(df[col].isnull().sum())
        entry: dict = {
            "column": col,
            "dtype": str(df[col].dtype),
            "count": int(df[col].count()),
            "null_count": null_count,
            "mean": _safe_float(df[col].mean()) if is_numeric else None,
            "min": _safe_float(df[col].min()) if is_numeric else None,
            "max": _safe_float(df[col].max()) if is_numeric else None,
        }
        columns.append(entry)

    data = {"total_rows": total_rows, "total_cols": total_cols, "columns": columns}
    return _resp(200, True, "Column statistics generated successfully", data)


def get_correlation_matrix(db: Session, file_id: uuid_pkg.UUID) -> tuple:
    """Compute the pairwise correlation matrix for all numeric columns in a CSV.

    Returns a dict with:
    - ``columns``: ordered list of column names included in the matrix
    - ``matrix``: NxN list of floats (NaN serialised as ``null``)

    Non-numeric and constant columns (std == 0) are silently excluded so that
    the heatmap only shows meaningful relationships.
    """
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

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return _resp(200, True, "No numeric columns found", {"columns": [], "matrix": []})

    corr = numeric_df.corr()
    # Convert the DataFrame to a plain list-of-lists; NaN becomes None (JSON null)
    columns = corr.columns.tolist()
    matrix = [[None if pd.isna(v) else round(float(v), 6) for v in row] for row in corr.to_numpy()]
    return _resp(200, True, "Correlation matrix computed successfully", {"columns": columns, "matrix": matrix})


def get_file_data(db: Session, file_id: uuid_pkg.UUID, offset: int = 0, limit: int = 100) -> tuple:
    """Read and return a paginated preview of a CSV file.

    The response includes:
    - ``rows``: list of records for the current page
    - ``columns``: ordered list of column names
    - ``pagination``: total/offset/limit metadata
    """
    file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
    if not file:
        return _resp(400, False, "Unable to open file")
    if file.file_type != "csv":
        return _resp(400, False, "Only CSV files support row preview")

    file_path = _get_file_path(file)
    try:
        # Cache columns from disk if missing in DB.
        if file.columns is None:
            header_df = pd.read_csv(file_path, nrows=0)
            file.columns = list(header_df.columns)
            db.add(file)
            db.commit()
        columns = file.columns or []

        # Use cached row_count when available; otherwise count once and cache.
        if file.row_count is None:
            file.row_count = sum(chunk.shape[0] for chunk in pd.read_csv(file_path, chunksize=10_000))
            db.add(file)
            db.commit()
        total_rows = file.row_count or 0

        # Read only the requested page rows (keep header row intact).
        skiprows = range(1, offset + 1) if offset > 0 else None
        df = pd.read_csv(file_path, skiprows=skiprows, nrows=limit)
    except FileNotFoundError:
        return _resp(500, False, f"File not found: {file_path}")
    except pd.errors.ParserError as e:
        logger.exception("CSV parsing error: %s", str(e))
        return _resp(500, False, f"Error reading CSV: {e}")
    except Exception as e:
        logger.exception("Error reading file: %s", str(e))
        return _resp(500, False, f"Error reading CSV: {e}")

    rows = df.to_dict(orient="records")
    data = {
        "rows": rows,
        "columns": columns,
        "pagination": {"total": total_rows, "offset": offset, "limit": limit},
    }
    return _resp(200, True, "Data sent successfully", data)


# Transformation handler functions
def _handle_one_hot_encoding(df: pd.DataFrame, feature: str, params: dict = None) -> pd.DataFrame:
    """Apply one-hot encoding to a categorical column."""
    return pd.get_dummies(df, columns=[feature])


def _handle_categorical_to_numerical(df: pd.DataFrame, feature: str, params: dict = None) -> pd.DataFrame:
    """Convert categorical values to numerical codes."""
    df[feature] = pd.Categorical(df[feature]).codes
    return df


def _handle_drop_column(df: pd.DataFrame, feature: str, params: dict = None) -> pd.DataFrame:
    """Drop a column from the dataframe."""
    return df.drop(columns=[feature])


def _handle_min_max_normalization(df: pd.DataFrame, feature: str, params: dict = None) -> pd.DataFrame:
    """Apply min-max normalization to a numeric column."""
    col_min = df[feature].min()
    col_max = df[feature].max()
    df[feature] = 0.0 if np.isclose(col_min, col_max) else (df[feature] - col_min) / (col_max - col_min)
    return df


def _handle_z_score_standardization(df: pd.DataFrame, feature: str, params: dict = None) -> pd.DataFrame:
    """Apply z-score standardization to a numeric column."""
    std = df[feature].std()
    df[feature] = 0.0 if std == 0 else (df[feature] - df[feature].mean()) / std
    return df


def _handle_log_transform(df: pd.DataFrame, feature: str, params: dict = None) -> pd.DataFrame:
    """Apply log transformation to a numeric column."""
    s = df[feature]
    if (s < -1).any():
        logger.warning(
            "Log Transform skipped for column '%s': %d value(s) below -1",
            feature,
            int((s < -1).sum()),
        )
    else:
        df[feature] = np.log1p(s)
    return df


def _handle_fill_missing_values(df: pd.DataFrame, feature: str, params: dict = None) -> pd.DataFrame:
    """Fill missing values in a column using specified strategy."""
    strategy = (params or {}).get("strategy", "mean")
    if strategy == "median":
        df[feature] = df[feature].fillna(df[feature].median())
    elif strategy == "mode":
        mode_vals = df[feature].mode()
        df[feature] = df[feature].fillna(mode_vals[0] if not mode_vals.empty else df[feature].mean())
    else:
        df[feature] = df[feature].fillna(df[feature].mean())
    return df


# Dispatch dictionary mapping transformation names to handler functions
_TRANSFORMATION_HANDLERS: dict[str, Callable] = {
    "One Hot Encoding": _handle_one_hot_encoding,
    "Categorical to Numerical": _handle_categorical_to_numerical,
    "Drop Column": _handle_drop_column,
    "Min-Max Normalization": _handle_min_max_normalization,
    "Z-score Standardization": _handle_z_score_standardization,
    "Log Transform": _handle_log_transform,
    "Fill Missing Values": _handle_fill_missing_values,
}

# Derived automatically — no manual sync needed
_VALID_TRANSFORMATIONS = set(_TRANSFORMATION_HANDLERS.keys())


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
            # Handle both dict-like and object attribute access
            if hasattr(t, "transformation"):
                name = t.transformation
                feature = t.feature
                params = getattr(t, "params", None)
            else:
                name = t.get("transformation")
                feature = t.get("feature")
                params = t.get("params")

            if name.casefold() not in {t.casefold() for t in _VALID_TRANSFORMATIONS}:
                return _resp(
                    422,
                    False,
                    f"Unsupported transformation '{name}'. Valid options: {sorted(_VALID_TRANSFORMATIONS)}",
                )
            if feature not in df.columns:
                return _resp(
                    422,
                    False,
                    f"Column '{feature}' not found. Available columns: {list(df.columns)}",
                )

        for t in transformations:
            # Handle both dict-like and object attribute access
            if hasattr(t, "transformation"):
                name = t.transformation
                feature = t.feature
                params = getattr(t, "params", None)
            else:
                name = t.get("transformation")
                feature = t.get("feature")
                params = t.get("params")

            # Find the matching transformation name (case-insensitive)
            actual_name = next(
                (valid_name for valid_name in _VALID_TRANSFORMATIONS if valid_name.casefold() == name.casefold()), None
            )

            if actual_name and actual_name in _TRANSFORMATION_HANDLERS:
                handler = _TRANSFORMATION_HANDLERS[actual_name]
                df = handler(df, feature, params)

        df.to_csv(file_path, index=False)
        return _resp(200, True, "Dataset preprocessed successfully")
    except pd.errors.ParserError as e:
        logger.exception("CSV parsing error: %s", str(e))
        return _resp(500, False, f"Error parsing CSV data: {e}")
    except Exception as e:
        logger.exception("Error preprocessing data: %s", str(e))
        return _resp(500, False, f"Error preprocessing data: {e}")
