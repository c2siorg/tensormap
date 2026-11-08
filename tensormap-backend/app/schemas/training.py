"""Request/response schemas for the training-job API."""

import uuid as uuid_pkg
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.shared.enums import Optimizer


class TrainingStartRequest(BaseModel):
    """Request body for starting a training run.

    Only ``model_name`` is required; hyperparameters default to whatever was
    saved on the model via the training-config endpoint, so existing callers
    keep working. Any field provided here overrides the stored value for this
    run (recorded in the job's ``hyperparams``).
    """

    model_name: str = Field(min_length=1)
    project_id: uuid_pkg.UUID | None = None
    optimizer: Optimizer | None = None
    lr: float | None = Field(default=None, gt=0)
    epochs: int | None = Field(default=None, gt=0)
    batch_size: int | None = Field(default=None, gt=0)


class TrainingJobResponse(BaseModel):
    """Full detail for a single training job."""

    job_id: str
    model_id: int
    status: str
    hyperparams: dict | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    latest_metrics: dict | None = None
    reproducibility_metadata: "ReproducibilitySnapshot | None" = None


class TrainingJobSummary(BaseModel):
    """Lightweight job descriptor for list views."""

    job_id: str
    model_id: int
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None


class DatasetProvenance(BaseModel):
    file_id: str
    filename: str | None = None
    dataset_type: Literal["csv", "zip"]
    fingerprint_algorithm: Literal["sha256"]
    fingerprint: str
    fingerprint_scope: Literal["file_bytes", "image_tree_manifest_v1"]
    size_bytes: int


class TrainingConfigurationSnapshot(BaseModel):
    problem_type_id: int | None = None
    problem_type: str | None = None
    target_column: str | None = None
    feature_columns: list[str] | None = None
    class_names: list[str] | None = None
    model_architecture: dict[str, Any] | None = None
    architecture_sha256: str | None = None
    optimizer: Any = None
    loss: Any = None
    metrics: list[str] = Field(default_factory=list)
    epochs: int | None = None
    batch_size: int | None = None
    training_split_percent: float | None = None
    holdout_split_percent: float | None = None
    requested_training_split_percent: float | None = None
    requested_holdout_split_percent: float | None = None
    actual_training_rows: int | None = None
    actual_holdout_rows: int | None = None
    actual_training_split_percent: float | None = None
    actual_holdout_split_percent: float | None = None
    holdout_role: Literal["validation", "evaluation"] | None = None
    configured_hyperparameters: dict[str, Any] = Field(default_factory=dict)
    effective_hyperparameters: dict[str, Any] = Field(default_factory=dict)


class RandomnessSnapshot(BaseModel):
    configured_seed: int | None = None
    effective_seeds: dict[str, int | None] = Field(default_factory=dict)
    applied_seeds: list[str] = Field(default_factory=list)
    global_framework_seed_applied: bool = False
    deterministic_execution_guaranteed: bool = False


class RuntimeVersions(BaseModel):
    python: str
    tensorflow: str | None = None
    keras: str | None = None
    numpy: str | None = None
    pandas: str | None = None


class ReproducibilitySnapshot(BaseModel):
    # Version 1 snapshots remain readable; new snapshots include realized split statistics.
    schema_version: Literal[1, 2] = 2
    job_id: str
    captured_at: datetime
    dataset: DatasetProvenance
    training_configuration: TrainingConfigurationSnapshot
    randomness: RandomnessSnapshot
    runtime_versions: RuntimeVersions


TrainingJobResponse.model_rebuild()
