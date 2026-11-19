"""Schemas for model architecture profiling and parameter estimation."""

from typing import Any

from pydantic import BaseModel, Field


class LayerProfileDetail(BaseModel):
    layer_name: str
    layer_type: str
    trainable_params: int
    non_trainable_params: int
    total_params: int
    param_memory_kb: float
    output_shape: list[Any] | None = None


class ArchitecturalWarning(BaseModel):
    category: str
    layer_name: str
    message: str


class ModelComplexityProfile(BaseModel):
    total_layers: int
    total_params: int
    trainable_params: int
    non_trainable_params: int
    param_memory_mb: float
    estimated_activation_memory_mb: float
    layers: list[LayerProfileDetail]
    warnings: list[ArchitecturalWarning] = Field(default_factory=list)
