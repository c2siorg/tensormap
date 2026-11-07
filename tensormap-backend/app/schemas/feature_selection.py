"""Request schemas for tabular feature selection."""

from typing import Literal

from pydantic import BaseModel, Field


class FeatureAnalysisRequest(BaseModel):
    """Request body for scoring dataset features against a target."""

    target: str = Field(min_length=1)
    task: Literal["auto", "classification", "regression"] = "auto"


class FeatureSelectionApplyRequest(BaseModel):
    """Request body for removing selected features from a dataset."""

    target: str = Field(min_length=1)
    columns_to_drop: list[str] = Field(min_length=1)
