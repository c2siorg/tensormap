"""Read-only quality analysis for uploaded training datasets."""

import uuid as uuid_pkg
from pathlib import Path
from typing import Any

import pandas as pd
from sqlmodel import Session, select

from app.config import get_settings
from app.models import DataFile
from app.shared.enums import ProblemType
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def _response(status_code: int, success: bool, message: str, data: Any = None) -> tuple:
    """Build the standard API response tuple."""
    return {"success": success, "message": message, "data": data}, status_code


def analyze_dataset_quality(
    db: Session,
    file_id: uuid_pkg.UUID,
    target_field: str | None = None,
    problem_type_id: int | None = None,
    training_split: float | None = None,
) -> tuple:
    """Analyze a CSV without changing it, using the same parser as training.

    Missing values follow pandas' existing ``read_csv`` defaults, which are
    also used by training. Zero and other non-null values remain valid values.
    """
    file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
    if file is None:
        return _response(404, False, "Dataset file was not found")

    if file.file_type != "csv":
        info = {
            "status": "not_applicable",
            "can_train": True,
            "summary": {"rows": file.row_count or 0, "columns": len(file.columns or []), "missing_values": 0},
            "issues": [
                {
                    "code": "tabular_analysis_not_applicable",
                    "severity": "info",
                    "columns": [],
                    "blocking": False,
                    "message": (
                        "Tabular quality checks do not apply to image archive datasets. "
                        "The image loader will validate this dataset when training starts."
                    ),
                }
            ],
            "limitations": [],
        }
        return _response(200, True, "Dataset quality report generated", info)

    settings = get_settings()
    upload_folder = Path(settings.upload_folder).resolve()
    path = (upload_folder / file.disk_name).resolve()
    if path == upload_folder or upload_folder not in path.parents:
        return _response(400, False, "Could not analyze this CSV. Check that the uploaded file is valid.")
    try:
        if path.stat().st_size > settings.max_content_length:
            return _response(413, False, "Dataset exceeds the configured upload size limit and cannot be analyzed.")
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame()
    except (pd.errors.ParserError, OSError, UnicodeDecodeError, MemoryError, ValueError):
        logger.exception("Could not analyze uploaded CSV for file_id=%s", file_id)
        return _response(
            400,
            False,
            "Could not analyze this CSV. Check that the uploaded file is a valid, readable CSV.",
        )

    rows = len(frame.index)
    columns = len(frame.columns)
    issues: list[dict[str, Any]] = []

    def add_issue(
        code: str,
        severity: str,
        message: str,
        affected_columns: list[str] | None = None,
        blocking: bool = False,
        **details: Any,
    ) -> None:
        issue = {
            "code": code,
            "severity": severity,
            "columns": affected_columns or [],
            "blocking": blocking,
            "message": message,
        }
        issue.update(details)
        issues.append(issue)

    if rows == 0:
        add_issue(
            "empty_dataset",
            "error",
            "This dataset has no data rows. Add at least one row before training.",
            blocking=True,
        )
    if columns == 0:
        add_issue(
            "empty_dataset",
            "error",
            "This dataset has no columns. Upload a CSV with a header and data columns.",
            blocking=True,
        )

    missing_by_column: list[tuple[str, int, float]] = []
    empty_columns: list[str] = []
    constant_columns: list[dict[str, Any]] = []
    missing_total = 0
    for column in frame.columns:
        values = frame[column]
        missing_count = int(values.isna().sum())
        missing_total += missing_count
        if missing_count:
            percentage = (missing_count / rows * 100) if rows else 0.0
            missing_by_column.append((str(column), missing_count, percentage))
            add_issue(
                "missing_values",
                "warning",
                (
                    f"{missing_count} of {rows} rows ({percentage:.1f}%) have a missing value in '{column}'. "
                    "Training currently drops rows containing missing values."
                ),
                [str(column)],
                count=missing_count,
                percentage=round(percentage, 2),
            )

        distinct_count = int(values.nunique(dropna=True))
        if distinct_count == 0:
            empty_columns.append(str(column))
            add_issue(
                "empty_column",
                "warning",
                f"Column '{column}' has no usable values.",
                [str(column)],
                distinct_count=0,
            )
        elif distinct_count == 1:
            constant_columns.append({"column": str(column), "distinct_count": distinct_count})
            add_issue(
                "constant_column",
                "warning",
                f"Column '{column}' has only one distinct non-missing value and may provide no useful variation.",
                [str(column)],
                distinct_count=distinct_count,
            )

    usable_rows = int((~frame.isna().any(axis=1)).sum()) if columns else 0
    if rows and usable_rows == 0:
        add_issue(
            "no_usable_rows",
            "error",
            "Training removes rows with missing values, leaving no usable rows. Resolve missing data before training.",
            blocking=True,
        )
    if training_split is not None and 0 < training_split < 100 and usable_rows:
        training_rows = int(usable_rows * training_split / 100)
        evaluation_rows = usable_rows - training_rows
        if training_rows == 0 or evaluation_rows == 0:
            add_issue(
                "empty_training_split",
                "error",
                "The selected train/test split leaves no rows in one partition. Add usable rows or adjust the split.",
                blocking=True,
                training_rows=training_rows,
                evaluation_rows=evaluation_rows,
            )

    target_required = problem_type_id != int(ProblemType.IMAGE_CLASSIFICATION)
    selected_target = target_field if isinstance(target_field, str) else ""
    if target_required and not selected_target.strip():
        add_issue("target_required", "error", "Select a target column before training.", blocking=True)
    elif selected_target and selected_target not in frame.columns:
        add_issue(
            "target_not_found",
            "error",
            f"Target column '{selected_target}' was not found in this dataset.",
            [selected_target],
            blocking=True,
        )
    elif selected_target and selected_target in frame.columns:
        target_values = frame[selected_target]
        target_distinct = int(target_values.nunique(dropna=True))
        if target_distinct == 0:
            add_issue(
                "empty_target",
                "error",
                f"Target column '{selected_target}' has no usable values.",
                [selected_target],
                blocking=True,
            )
        elif problem_type_id == int(ProblemType.CLASSIFICATION) and target_distinct == 1:
            add_issue(
                "single_class_target",
                "error",
                (
                    f"Classification target '{selected_target}' has only one distinct non-missing class. "
                    "Provide at least two classes before training."
                ),
                [selected_target],
                blocking=True,
                distinct_count=target_distinct,
            )
        if columns <= 1:
            add_issue(
                "no_feature_columns",
                "error",
                "The target is the only column. Add at least one feature column before training.",
                [selected_target],
                blocking=True,
            )

    duplicate_header_limitation = (
        "Duplicate CSV headers may be normalized by pandas during parsing, so this report cannot reliably "
        "identify original duplicate names."
    )
    blocking = any(issue["blocking"] for issue in issues)
    status = "blocked" if blocking else "warnings" if issues else "clean"
    report = {
        "status": status,
        "can_train": not blocking,
        "summary": {
            "rows": rows,
            "usable_rows": usable_rows,
            "columns": columns,
            "missing_values": missing_total,
            "affected_missing_columns": len(missing_by_column),
            "constant_columns": len(constant_columns),
            "empty_columns": len(empty_columns),
        },
        "issues": issues,
        "limitations": [duplicate_header_limitation],
    }
    return _response(200, True, "Dataset quality report generated", report)
