"""GAP-2: Data quality and preprocessing suggestion endpoints."""

import uuid as uuid_pkg

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.database import get_db
from app.models.data import DataFile

router = APIRouter(tags=["Data Quality"])


@router.get("/file/{file_id}/quality-report")
def get_quality_report(file_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
    """Return the full data quality report for an uploaded file."""
    file = db.get(DataFile, file_id)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")

    return {
        "file_id": str(file.id),
        "file_name": file.file_name,
        "file_type": file.file_type,
        "file_size_mb": file.file_size_mb,
        "row_count": file.row_count,
        "upload_duration_seconds": file.upload_duration_seconds,
        "quality": {
            "score": file.data_quality_score,
            "validation_status": file.validation_status,
            "validation_messages": file.validation_messages or [],
        },
        "missing_values": {
            "has_missing_values": file.has_missing_values,
            "missing_value_count": file.missing_value_count,
            "missing_columns": file.missing_columns or [],
            "column_nulls": file.column_nulls or {},
        },
        "duplicates": {
            "has_duplicates": file.has_duplicates,
            "duplicate_row_count": file.duplicate_row_count,
        },
        "class_balance": {
            "class_distribution": file.class_distribution,
            "is_imbalanced": file.is_imbalanced,
            "imbalance_ratio": file.imbalance_ratio,
            "note": "Class distribution computed on last column — verify this is your target column",
        },
    }


@router.get("/file/{file_id}/columns-info")
def get_columns_info(file_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
    """Return column-level type and null information for an uploaded file."""
    file = db.get(DataFile, file_id)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")

    return {
        "file_id": str(file.id),
        "file_name": file.file_name,
        "columns": file.columns or [],
        "column_dtypes": file.column_dtypes or {},
        "column_nulls": file.column_nulls or {},
        "numeric_columns": file.numeric_columns or [],
        "categorical_columns": file.categorical_columns or [],
    }


@router.post("/file/{file_id}/suggest-preprocessing")
def suggest_preprocessing(file_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
    """Return actionable preprocessing suggestions based on data quality metrics."""
    file = db.get(DataFile, file_id)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")

    suggestions = []

    if file.is_imbalanced and file.imbalance_ratio:
        suggestions.append(
            {
                "type": "class_imbalance",
                "severity": "high" if file.imbalance_ratio > 10 else "medium",
                "message": f"Dataset is imbalanced ({file.imbalance_ratio}x ratio). "
                "Consider class_weight='balanced' or SMOTE oversampling.",
            }
        )

    if file.has_missing_values and file.missing_value_count > 0:
        pct = round(
            file.missing_value_count / max((file.row_count or 1) * len(file.columns or [1]), 1) * 100,
            1,
        )
        severity = "high" if pct > 20 else "medium" if pct > 5 else "low"
        suggestions.append(
            {
                "type": "missing_values",
                "severity": severity,
                "message": f"{file.missing_value_count} missing values ({pct}%) in columns: "
                f"{', '.join(file.missing_columns or [])}. "
                "Consider imputing with mean/median or dropping affected rows.",
            }
        )

    if file.has_duplicates and file.duplicate_row_count > 0:
        suggestions.append(
            {
                "type": "duplicates",
                "severity": "low",
                "message": f"{file.duplicate_row_count} duplicate rows found. "
                "Consider dropping duplicates before training.",
            }
        )

    if file.row_count and file.row_count < 100:
        suggestions.append(
            {
                "type": "small_dataset",
                "severity": "high",
                "message": f"Only {file.row_count} rows — model may underfit. "
                "Consider collecting more data or using data augmentation.",
            }
        )

    return {
        "file_id": str(file.id),
        "file_name": file.file_name,
        "quality_score": file.data_quality_score,
        "total_suggestions": len(suggestions),
        "suggestions": suggestions,
    }
