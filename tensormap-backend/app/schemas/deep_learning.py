"""Request schemas for model validation, code generation, and training endpoints."""

import uuid as uuid_pkg
from typing import Any

from pydantic import BaseModel, Field


# --- "code" sub-models ---
class DatasetConfig(BaseModel):
    """Dataset parameters for model training (file, target, split)."""

    file_id: uuid_pkg.UUID
    target_field: str | None = None  # null for image classification
    training_split: float = Field(gt=0, le=100)


class DLModelConfig(BaseModel):
    """Training hyperparameters (optimizer, metric, epochs)."""

    model_name: str = Field(min_length=1)
    optimizer: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    epochs: int = Field(gt=0)


class CodeConfig(BaseModel):
    """Combined dataset and model configuration for code generation."""

    dataset: DatasetConfig
    dl_model: DLModelConfig
    problem_type_id: int = Field(ge=1, le=3)


# --- "model" (ReactFlow graph) sub-models ---
class NodeData(BaseModel):
    """Per-node parameters from the ReactFlow canvas."""
    
    params: dict[str, Any]  # varies per node type; validated downstream by model_generation + TensorFlow
    registry: dict[str, Any] | None = None  # <--- THE GOLDEN TICKET

class GraphNode(BaseModel):
    """A single node in the ReactFlow model graph."""

    id: str
    type: str
    data: NodeData


class GraphEdge(BaseModel):
    """A directed edge between two nodes in the model graph."""

    source: str
    target: str


class ModelGraph(BaseModel):
    """Complete ReactFlow graph with nodes, edges, and a model name."""

    nodes: list[GraphNode] = Field(min_length=1)
    edges: list[GraphEdge]
    model_name: str = Field(min_length=1)


# --- Top-level request schemas ---
class ModelValidateRequest(BaseModel):
    """Request body for validating and saving a model from the canvas."""

    model: ModelGraph
    code: CodeConfig
    project_id: uuid_pkg.UUID | None = None


class ModelNameRequest(BaseModel):
    """Request body referencing a saved model by name."""

    model_name: str
    project_id: uuid_pkg.UUID | None = None


class ModelSaveRequest(BaseModel):
    """Request body for saving model architecture only (no training config)."""

    model: ModelGraph
    model_name: str = Field(min_length=1)
    project_id: uuid_pkg.UUID | None = None


class TrainingConfigRequest(BaseModel):
    """Request body for setting training configuration on a saved model."""

    model_name: str = Field(min_length=1)
    file_id: uuid_pkg.UUID
    target_field: str | None = None
    training_split: float = Field(gt=0, le=100)
    problem_type_id: int = Field(ge=1, le=3)
    optimizer: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    epochs: int = Field(gt=0)
    project_id: uuid_pkg.UUID | None = None
