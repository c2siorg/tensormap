import os
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


def _paginated_resp(data: list, pagination: dict) -> tuple:
    """Build a standard API response tuple for paginated data."""
    body = {
        "success": True,
        "message": "Data sent successfully",
        "data": data,
        "pagination": pagination,
    }
    return body, 200


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
        select(
            DataProcess.file_id,
            DataFile.file_name,
            DataFile.file_type,
            DataProcess.target,
        )
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
    body = {
        "success": True,
        "message": "Target fields of all files received successfully",
        "data": data,
    }
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
    return _resp(
        200,
        True,
        "Correlation matrix computed successfully",
        {"columns": columns, "matrix": matrix},
    )


def get_file_data(db: Session, file_id: uuid_pkg.UUID, page: int = 1, page_size: int = 50) -> tuple:
    """Read and return the paginated contents of a CSV file as JSON."""
    file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
    if not file:
        return _resp(400, False, "Unable to open file")

    file_path = _get_file_path(file)

    try:
        with open(file_path, "rb") as f:
            total_rows = sum(1 for _ in f) - 1
        total_rows = max(total_rows, 0)
    except FileNotFoundError:
        return _resp(500, False, f"File not found: {file_path}")
    except Exception as e:
        return _resp(500, False, f"Error reading file count: {e}")

    total_pages = (total_rows + page_size - 1) // page_size if page_size > 0 else 0

    if total_rows > 0 and page > total_pages:
        return _resp(400, False, f"Page {page} exceeds total pages ({total_pages})")

    if total_rows == 0:
        return _paginated_resp([], {"page": page, "page_size": page_size, "total_rows": 0, "total_pages": 0})

    start_idx = (page - 1) * page_size
    skip = list(range(1, start_idx + 1)) if start_idx > 0 else None

    try:
        df_page = pd.read_csv(file_path, skiprows=skip, nrows=page_size)
    except FileNotFoundError:
        return _resp(500, False, f"File not found: {file_path}")
    except pd.errors.ParserError as e:
        logger.exception("CSV parsing error: %s", str(e))
        return _resp(500, False, f"Error reading CSV: {e}")
    except Exception as e:
        logger.exception("Error reading file: %s", str(e))
        return _resp(500, False, f"Error reading CSV: {e}")

    # For empty or any dataframe slice, to_dict will convert to list of plain dict elements avoiding json strings
    data_list = df_page.to_dict(orient="records")

    return _paginated_resp(
        data_list, {"page": page, "page_size": page_size, "total_rows": total_rows, "total_pages": total_pages}
    )


def _one_hot_encode(df: pd.DataFrame, col: str, _params: dict | None) -> pd.DataFrame:
    return pd.get_dummies(df, columns=[col], dtype=int)


def _categorical_to_numerical(df: pd.DataFrame, col: str, _params: dict | None) -> pd.DataFrame:
    # Preserve integer dtype so downstream consumers see a proper int column.
    # pd.Categorical.codes returns int with -1 for NaNs — keep that convention
    # rather than promoting to float just to hold NaN.
    df[col] = pd.Categorical(df[col]).codes
    return df


def _drop_column(df: pd.DataFrame, col: str, _params: dict | None) -> pd.DataFrame:
    return df.drop(columns=[col])


def _min_max_normalize(df: pd.DataFrame, col: str, _params: dict | None) -> pd.DataFrame:
    col_min = df[col].min()
    col_max = df[col].max()
    df[col] = 0.0 if np.isclose(col_min, col_max) else (df[col] - col_min) / (col_max - col_min)
    return df


def _zscore_standardize(df: pd.DataFrame, col: str, _params: dict | None) -> pd.DataFrame:
    mean = df[col].mean()
    std = df[col].std()
    df[col] = 0.0 if np.isclose(std, 0) else (df[col] - mean) / std
    return df


def _log_transform(df: pd.DataFrame, col: str, _params: dict | None) -> pd.DataFrame:
    s = df[col]
    invalid = int((s < -1).sum())
    if invalid:
        raise ValueError(f"Log Transform: {invalid} value(s) below -1 in column '{col}'")
    df[col] = np.log1p(s)
    return df


def _fill_missing(df: pd.DataFrame, col: str, params: dict | None) -> pd.DataFrame:
    strategy = (params or {}).get("strategy", "mean")
    if strategy == "median":
        df[col] = df[col].fillna(df[col].median())
    elif strategy == "mode":
        mode_vals = df[col].mode()
        fill_val = mode_vals[0] if not mode_vals.empty else df[col].median()
        df[col] = df[col].fillna(fill_val)
    elif strategy == "mean":
        df[col] = df[col].fillna(df[col].mean())
    else:
        raise ValueError(f"Unknown fill strategy '{strategy}'. Valid: 'mean', 'median', 'mode'")
    return df


# Single source of truth: maps transformation name → handler function.
# Adding a new transformation only requires adding one entry here.
# This is the ONLY place to register new transformations.
_TRANSFORMATION_REGISTRY: dict[str, Callable[..., pd.DataFrame]] = {
    "One Hot Encoding": _one_hot_encode,
    "Categorical to Numerical": _categorical_to_numerical,
    "Drop Column": _drop_column,
    "Min-Max Normalization": _min_max_normalize,
    "Z-score Standardization": _zscore_standardize,
    "Log Transform": _log_transform,
    "Fill Missing Values": _fill_missing,
}


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
            if t.transformation not in _TRANSFORMATION_REGISTRY:
                return _resp(
                    422,
                    False,
                    f"Unknown transformation '{t.transformation}'. Valid options: {sorted(_TRANSFORMATION_REGISTRY)}",
                )
            if t.feature not in df.columns:
                return _resp(
                    422,
                    False,
                    f"Column '{t.feature}' not found. Available columns: {list(df.columns)}",
                )

        for t in transformations:
            if t.feature not in df.columns:
                return _resp(
                    422,
                    False,
                    f"Column '{t.feature}' not found (may have been removed by a prior transformation)",
                )
            handler = _TRANSFORMATION_REGISTRY[t.transformation]
            df = handler(df, t.feature, t.params)

        df.to_csv(file_path, index=False)
        return _resp(200, True, "Dataset preprocessed successfully")
    except ValueError as e:
        return _resp(422, False, str(e))
    except pd.errors.ParserError as e:
        logger.exception("CSV parsing error: %s", str(e))
        return _resp(500, False, f"Error parsing CSV data: {e}")
    except Exception as e:
        logger.exception("Error preprocessing data: %s", str(e))
        return _resp(500, False, f"Error preprocessing data: {e}")


def augment_image_service(
    db: Session,
    file_id: uuid_pkg.UUID,
    technique: str = "flip_horizontal",
) -> tuple:
    """Apply image augmentation techniques to generate synthetic variants.

    Supported techniques:
    - flip_horizontal: Mirror image along vertical axis
    - flip_vertical: Mirror image along horizontal axis
    - rotate_90: Rotate image by 90 degrees
    - brightness: Adjust brightness by 20%
    - zoom: Zoom to 90% then resize
    - gaussian_noise: Add Gaussian noise
    - random_crop: Crop to 85% then resize
    """
    file_record = db.get(DataFile, file_id)
    if not file_record:
        return _resp(404, False, "Image file not found")

    file_path = file_record.file_path
    if not file_path or not os.path.exists(file_path):
        return _resp(404, False, "Image file not found on disk")

    settings = get_settings()
    output_dir = os.path.join(settings.UPLOAD_DIRECTORY, f"augmented_{file_id}")
    os.makedirs(output_dir, exist_ok=True)

    supported_formats = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in supported_formats:
        return _resp(400, False, f"Unsupported image format: {ext}")

    try:
        from PIL import Image, ImageEnhance

        original = Image.open(file_path)

        if technique == "flip_horizontal":
            augmented = original.transpose(Image.FLIP_LEFT_RIGHT)
        elif technique == "flip_vertical":
            augmented = original.transpose(Image.FLIP_TOP_BOTTOM)
        elif technique == "rotate_90":
            augmented = original.rotate(90, expand=True)
        elif technique == "brightness":
            enhancer = ImageEnhance.Brightness(original)
            augmented = enhancer.enhance(1.2)
        elif technique == "zoom":
            width, height = original.size
            new_width = int(width * 0.9)
            new_height = int(height * 0.9)
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            cropped = original.crop((left, top, left + new_width, top + new_height))
            augmented = cropped.resize((width, height), Image.LANCZOS)
        elif technique == "gaussian_noise":
            np_img = np.array(original.convert("RGB")).astype(np.float32) / 255.0
            noise = np.random.normal(0, 0.05, np_img.shape)
            np_img = np.clip(np_img + noise, 0, 1)
            augmented = Image.fromarray((np_img * 255).astype(np.uint8))
        elif technique == "random_crop":
            width, height = original.size
            crop_size = int(min(width, height) * 0.85)
            left = np.random.randint(0, width - crop_size + 1)
            top = np.random.randint(0, height - crop_size + 1)
            cropped = original.crop((left, top, left + crop_size, top + crop_size))
            augmented = cropped.resize((width, height), Image.LANCZOS)
        else:
            return _resp(400, False, f"Unknown technique: {technique}")

        output_path = os.path.join(output_dir, f"augmented_0{ext}")
        augmented.save(output_path)

        logger.info("Applied %s augmentation to file %s", technique, file_id)
        return _resp(200, True, f"Generated augmented image using {technique}", {"output_path": output_path})

    except ImportError:
        return _resp(500, False, "Pillow not installed")
    except Exception as e:
        logger.exception("Error augmenting image: %s", str(e))
        return _resp(500, False, f"Error augmenting image: {e}")
