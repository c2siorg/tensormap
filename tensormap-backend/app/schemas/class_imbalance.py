"""Schemas for dataset class imbalance inspection and cost-sensitive class weight calculation."""

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class ImbalanceSeverity(StrEnum):
    """Classification distribution imbalance severity tier."""

    BALANCED = "balanced"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class ClassDistributionItem(BaseModel):
    """Detailed count, frequency, and recommended cost-sensitive weight for a class label."""

    class_name: str
    count: int
    percentage: float = Field(..., description="Percentage of dataset represented by this class (0-100)")
    balanced_weight: float = Field(..., description="Heuristic class weight: N / (n_classes * N_c)")


class ClassImbalanceResponse(BaseModel):
    """Diagnostic scorecard for dataset target label balance."""

    file_id: UUID
    target_field: str
    total_samples: int
    n_classes: int
    imbalance_ratio: float = Field(..., description="Ratio of majority class count to minority class count")
    shannon_entropy: float = Field(..., description="Shannon entropy of the label distribution (bits)")
    normalized_entropy: float = Field(..., description="Entropy normalized to [0, 1] relative to uniform distribution")
    gini_impurity: float = Field(..., description="Gini impurity score (0 = pure, max = 1 - 1/k)")
    severity: ImbalanceSeverity
    is_imbalanced: bool
    recommended_strategy: str
    class_distribution: list[ClassDistributionItem]
