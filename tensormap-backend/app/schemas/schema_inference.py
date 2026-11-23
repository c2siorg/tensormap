"""Schemas for dataset semantic column type inference and feature recommendations."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SemanticType(StrEnum):
    """Semantic data category inferred beyond raw dataframe dtypes."""

    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    IDENTIFIER_KEY = "identifier_key"
    TEXT = "text"


class ColumnInferenceDetail(BaseModel):
    """Semantic analysis and pre-processing recommendations for a single column."""

    name: str
    raw_dtype: str
    semantic_type: SemanticType
    unique_count: int
    null_count: int
    sample_values: list[Any] = Field(default_factory=list)
    suggested_action: str


class SchemaInferenceResponse(BaseModel):
    """Full semantic schema summary and column-level guidance for a tabular dataset."""

    total_columns: int
    total_rows: int
    columns: list[ColumnInferenceDetail]
    semantic_counts: dict[str, int]
