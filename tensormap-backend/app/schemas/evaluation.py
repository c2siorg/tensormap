"""Pydantic schemas for holdout dataset batch evaluation and prediction."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class BatchEvaluationRequest(BaseModel):
    """Request payload to evaluate a trained model against an uploaded dataset."""

    file_id: UUID = Field(..., description="UUID of the uploaded dataset file to evaluate")
    target_field: str | None = Field(
        None, description="Target column name in the dataset (defaults to model's configured target)"
    )
    batch_size: int = Field(256, ge=1, le=2048, description="Batch size for model inference")


class TopErrorSample(BaseModel):
    """Detailed record of a misclassified or high-residual sample from evaluation."""

    index: int = Field(..., description="Zero-based row index in the evaluation dataset")
    features: dict[str, Any] = Field(..., description="Input feature values for this sample")
    actual_label: str = Field(..., description="Ground truth target value or class label")
    predicted_label: str = Field(..., description="Predicted target value or class label")
    confidence: float | None = Field(None, description="Prediction probability/confidence (classification only)")
    loss_or_residual: float = Field(..., description="Error magnitude: confidence of false class or absolute residual")


class BatchEvaluationResponse(BaseModel):
    """Comprehensive evaluation scorecard on the holdout dataset."""

    job_id: str
    file_id: str
    problem_type: str
    total_samples: int
    evaluated_at: str
    metrics: dict[str, Any]
    top_errors: list[TopErrorSample]


class BatchPredictionRequest(BaseModel):
    """Request payload to generate batch predictions on an unlabelled or test dataset."""

    file_id: UUID = Field(..., description="UUID of the dataset file to predict")
    batch_size: int = Field(256, ge=1, le=2048, description="Batch size for model inference")


class BatchPredictionResponse(BaseModel):
    """Summary and preview rows of generated batch predictions."""

    job_id: str
    file_id: str
    problem_type: str
    total_rows: int
    columns: list[str]
    preview_rows: list[dict[str, Any]]
