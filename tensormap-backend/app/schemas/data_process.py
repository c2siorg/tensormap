"""Request schemas for data processing endpoints."""

import uuid as uuid_pkg

from pydantic import BaseModel


class TargetAddRequest(BaseModel):
    """Request body for setting a dataset's target field."""

    target: str
    file_id: uuid_pkg.UUID


class TransformationItem(BaseModel):
    """A single column transformation (e.g., one-hot encoding, drop)."""

    transformation: str
    feature: str
    params: dict | None = None


class PreprocessRequest(BaseModel):
    """Request body for applying preprocessing transformations to a dataset."""

    transformations: list[TransformationItem]


class ClassDetail(BaseModel):
    """Class frequency, proportion, and balanced weight for a single class label."""

    class_name: str
    count: int
    proportion: float
    weight: float


class ClassImbalanceData(BaseModel):
    """Detailed class imbalance diagnostics and balanced class weights."""

    file_id: str
    target_field: str
    total_samples: int
    null_count: int
    num_classes: int
    imbalance_ratio: float
    majority_class: str
    minority_class: str
    shannon_entropy: float
    normalized_entropy: float
    gini_impurity: float
    severity: str
    recommendation: str
    class_weights: dict[str, float]
    classes: list[ClassDetail]


class ClassImbalanceResponse(BaseModel):
    """Response payload for class imbalance inspection endpoint."""

    success: bool
    message: str
    data: ClassImbalanceData | None = None
