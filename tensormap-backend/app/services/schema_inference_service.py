"""Service for inferring semantic column types and pre-processing recommendations on CSV datasets."""

import uuid as uuid_pkg
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sqlmodel import Session

from app.config import get_settings
from app.models import DataFile
from app.schemas.schema_inference import (
    ColumnInferenceDetail,
    SchemaInferenceResponse,
    SemanticType,
)
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

_BOOL_SET = {"0", "1", "0.0", "1.0", "true", "false", "yes", "no", "t", "f", "y", "n"}


def infer_series_semantic_type(series: pd.Series) -> tuple[SemanticType, str]:
    """Infers semantic data category and suggested action for a single pandas Series."""
    clean = series.dropna()
    if clean.empty:
        return SemanticType.TEXT, "drop_empty_column"

    n_unique = clean.nunique()
    col_name = str(series.name or "").lower().strip()

    # 1. Boolean check (strictly 2 unique values matching boolean literals, not constant columns)
    if pd.api.types.is_bool_dtype(clean):
        return SemanticType.BOOLEAN, "binary_encode"
    if n_unique == 2:
        str_vals = {str(v).strip().lower() for v in clean}
        if str_vals.issubset(_BOOL_SET):
            return SemanticType.BOOLEAN, "binary_encode"

    # 2. Identifier key check (exact 'id', '_id' suffix, or common key suffixes with boundary check)
    if (
        len(clean) >= 10
        and n_unique == len(clean)
        and (col_name in {"id", "uuid", "key", "index"} or col_name.endswith(("_id", "_uuid", "_key", "_index")))
    ):
        return SemanticType.IDENTIFIER_KEY, "drop_identifier"

    # 3. Datetime check
    if pd.api.types.is_datetime64_any_dtype(clean):
        return SemanticType.DATETIME, "extract_datetime_features"
    if pd.api.types.is_string_dtype(clean) or pd.api.types.is_object_dtype(clean):
        sample = clean.iloc[:30].astype(str).str.strip()
        # Avoid treating plain integers or short numbers (e.g. '2020', '12') as datetimes
        is_not_plain_num = not sample.str.replace(".", "", regex=False).str.isdigit().all()
        if is_not_plain_num and sample.str.len().mean() >= 6:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().mean() >= 0.8:
                        return SemanticType.DATETIME, "extract_datetime_features"
                except Exception:
                    pass

    # 4. Numeric continuous vs discrete check
    if pd.api.types.is_numeric_dtype(clean):
        if pd.api.types.is_float_dtype(clean) or n_unique > 25:
            return SemanticType.NUMERIC_CONTINUOUS, "standard_scale"
        return SemanticType.NUMERIC_DISCRETE, "label_encode" if n_unique > 5 else "one_hot_encode"

    # 5. Categorical vs free-form text
    if n_unique <= 30 or (n_unique / len(clean) <= 0.2):
        return SemanticType.CATEGORICAL, "one_hot_encode" if n_unique <= 8 else "label_encode"

    return SemanticType.TEXT, "text_embedding"


def infer_dataframe_schema(df: pd.DataFrame, total_rows: int | None = None) -> SchemaInferenceResponse:
    """Inspects all columns in a DataFrame and compiles the semantic schema report."""
    columns: list[ColumnInferenceDetail] = []
    counts: Counter[str] = Counter()

    for col in df.columns:
        s = df[col]
        sem_type, action = infer_series_semantic_type(s)
        counts[sem_type.value] += 1
        samples = [v for v in s.dropna().iloc[:3].tolist()]

        columns.append(
            ColumnInferenceDetail(
                name=str(col),
                raw_dtype=str(s.dtype),
                semantic_type=sem_type,
                unique_count=int(s.nunique(dropna=True)),
                null_count=int(s.isna().sum()),
                sample_values=samples,
                suggested_action=action,
            )
        )

    return SchemaInferenceResponse(
        total_columns=len(columns),
        total_rows=total_rows if total_rows is not None else len(df),
        columns=columns,
        semantic_counts=dict(counts),
    )


def infer_dataset_schema_service(
    db: Session,
    file_id: uuid_pkg.UUID,
    sample_size: int = 1000,
) -> tuple[dict[str, Any], int]:
    """Retrieves uploaded CSV dataset and runs semantic column type inference."""
    data_file = db.get(DataFile, file_id)
    if not data_file:
        return {"success": False, "message": "File not found", "data": None}, 404

    settings = get_settings()
    upload_folder = Path(settings.upload_folder).resolve()
    csv_path = (upload_folder / data_file.disk_name).resolve()
    if not csv_path.exists():
        return {"success": False, "message": "File data not found on disk", "data": None}, 404

    # Path traversal safety check
    try:
        csv_path.relative_to(upload_folder)
    except ValueError:
        return {"success": False, "message": "Invalid file path", "data": None}, 400

    # Determine true total row count
    actual_total_rows = data_file.row_count
    if not actual_total_rows:
        try:
            with open(csv_path, encoding="utf-8", errors="ignore") as f:
                actual_total_rows = max(0, sum(1 for _ in f) - 1)
        except OSError:
            actual_total_rows = sample_size

    try:
        df = pd.read_csv(csv_path, nrows=sample_size)
        if df.empty:
            return {"success": False, "message": "Dataset CSV is empty", "data": None}, 400
        schema = infer_dataframe_schema(df, total_rows=actual_total_rows)
        return {
            "success": True,
            "message": "Dataset schema inferred successfully",
            "data": schema.model_dump(),
        }, 200
    except pd.errors.EmptyDataError:
        return {"success": False, "message": "Dataset CSV contains no data", "data": None}, 400
    except pd.errors.ParserError:
        return {"success": False, "message": "Malformed CSV file could not be parsed", "data": None}, 400
    except Exception as e:
        logger.exception("Failed to infer schema for file_id=%s: %s", file_id, e)
        return {"success": False, "message": "Failed to analyze dataset schema", "data": None}, 500
