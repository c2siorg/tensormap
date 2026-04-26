"""GAP-2: Data quality analysis computed at upload time."""

import os

import numpy as np
import pandas as pd

from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def _calculate_quality_score(df: pd.DataFrame, imbalance_threshold: float = 3.0) -> int:
    """Score 0-100: penalise missing values, duplicates, and single-column datasets.

    Args:
        df: DataFrame to score
        imbalance_threshold: Ratio threshold for class imbalance (default 3.0)
    """
    if df.empty or len(df) == 0:
        return 0

    score = 100

    # Penalise missing values (up to -40)
    missing_pct = df.isnull().mean().mean() * 100
    score -= min(40, int(missing_pct * 2))

    # Penalise duplicates (up to -20)
    dup_pct = (len(df) - len(df.drop_duplicates())) / max(len(df), 1) * 100
    score -= min(20, int(dup_pct * 2))

    # Penalise tiny datasets (up to -10, gentler penalty)
    if len(df) < 50:
        score -= 10
    elif len(df) < 200:
        score -= 5

    # Penalise single column (up to -20)
    if len(df.columns) < 2:
        score -= 20

    return max(0, score)


def _calculate_imbalance_ratio(series: pd.Series) -> float | None:
    """Return max/min class count ratio. None if only one class or <2 samples.

    Returns:
        Imbalance ratio rounded to 2 decimals, or None if single class/insufficient data
    """
    if series.isna().all():
        return None
    counts = series.value_counts()
    if len(counts) < 2:
        return None
    return round(counts.max() / max(counts.min(), 1), 2)


def analyze_data_file(
    file_path: str,
    imbalance_threshold: float = 3.0,
    quality_score_threshold_valid: int = 80,
    quality_score_threshold_needs_prep: int = 50,
) -> dict:
    """Compute all quality metrics for a CSV file at upload time.

    Args:
        file_path: Path to CSV file
        imbalance_threshold: Ratio threshold for class imbalance (default 3.0)
        quality_score_threshold_valid: Score threshold for 'valid' status (default 80)
        quality_score_threshold_needs_prep: Score threshold for 'needs_preprocessing' (default 50)

    Returns:
        Dict with all quality metrics, or dict with 'error' key on failure
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            logger.warning("File %s is empty", file_path)
            return {"error": "File is empty", "analysis_status": "failed"}
    except pd.errors.ParserError as exc:
        logger.error("Could not parse CSV file %s: %s", file_path, exc)
        return {"error": f"CSV parsing error: {str(exc)}", "analysis_status": "failed"}
    except Exception as exc:
        logger.error("Could not analyze file %s: %s", file_path, exc)
        return {"error": f"Analysis failed: {str(exc)}", "analysis_status": "failed"}

    file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 4)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    missing_value_count = int(df.isnull().sum().sum())
    missing_columns = df.columns[df.isnull().any()].tolist()
    column_nulls = {col: int(n) for col, n in df.isnull().sum().items() if n > 0}
    column_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    duplicate_row_count = int(len(df) - len(df.drop_duplicates()))

    quality_score = _calculate_quality_score(df, imbalance_threshold=imbalance_threshold)

    # Attempt class distribution on last column as a proxy target
    # NOTE: This assumes last column is the target — user should verify in UI
    last_col = df.columns[-1] if len(df.columns) > 0 else None
    class_distribution = None
    imbalance_ratio = None
    is_imbalanced = False
    if last_col and df[last_col].nunique() <= 20:
        class_distribution = {str(k): int(v) for k, v in df[last_col].value_counts().items()}
        imbalance_ratio = _calculate_imbalance_ratio(df[last_col])
        is_imbalanced = imbalance_ratio is not None and imbalance_ratio > imbalance_threshold
        if class_distribution:
            logger.info(
                "Detected potential target column '%s' with %d classes",
                last_col,
                len(class_distribution),
            )

    # Validation messages
    messages = []
    if missing_value_count > 0:
        pct = round(missing_value_count / max(df.size, 1) * 100, 1)
        messages.append(f"{missing_value_count} missing values ({pct}%) across {len(missing_columns)} column(s)")
    if duplicate_row_count > 0:
        messages.append(f"{duplicate_row_count} duplicate rows detected")
    if is_imbalanced:
        messages.append(
            f"Potential target column '{last_col}' is imbalanced (ratio {imbalance_ratio}x) — "
            f"verify this is your target before training"
        )
    if len(df) < 100:
        messages.append("Dataset has fewer than 100 rows — model may underfit")

    if quality_score >= quality_score_threshold_valid:
        validation_status = "valid"
    elif quality_score >= quality_score_threshold_needs_prep:
        validation_status = "needs_preprocessing"
    else:
        validation_status = "invalid"

    # Log analysis summary
    logger.info(
        "Analysis complete for %s: score=%d, status=%s, missing=%d, duplicates=%d",
        file_path,
        quality_score,
        validation_status,
        missing_value_count,
        duplicate_row_count,
    )

    return {
        "analysis_status": "success",
        "file_size_mb": file_size_mb,
        "data_quality_score": quality_score,
        "has_missing_values": missing_value_count > 0,
        "missing_value_count": missing_value_count,
        "missing_columns": missing_columns,
        "column_nulls": column_nulls,
        "column_dtypes": column_dtypes,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "has_duplicates": duplicate_row_count > 0,
        "duplicate_row_count": duplicate_row_count,
        "class_distribution": class_distribution,
        "is_imbalanced": is_imbalanced,
        "imbalance_ratio": imbalance_ratio,
        "validation_status": validation_status,
        "validation_messages": messages,
    }
