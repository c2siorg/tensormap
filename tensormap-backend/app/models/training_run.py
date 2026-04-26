from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Column, DateTime, ForeignKey, String, func
from sqlmodel import Field, Relationship, SQLModel


class ModelTrainingRun(SQLModel, table=True):
    """Records every training run for a model, capturing metrics and config."""

    __tablename__ = "model_training_run"

    id: int | None = Field(default=None, primary_key=True)
    model_id: int = Field(
        sa_column=Column(ForeignKey("model_basic.id", ondelete="CASCADE"), index=True, nullable=False)
    )

    # Timing
    started_at: datetime = Field(sa_column=Column(DateTime, nullable=False))
    completed_at: datetime | None = Field(default=None, sa_column=Column(DateTime, nullable=True))
    duration_seconds: float | None = Field(default=None, nullable=True)

    # Config snapshot at time of training
    epochs_configured: int | None = Field(default=None, nullable=True)
    batch_size_configured: int | None = Field(default=None, nullable=True)
    training_split_configured: float | None = Field(default=None, nullable=True)
    optimizer: str | None = Field(default=None, max_length=50, nullable=True)
    loss_fn: str | None = Field(default=None, max_length=50, nullable=True)
    metric_name: str | None = Field(default=None, max_length=50, nullable=True)

    # Final results
    final_train_loss: float | None = Field(default=None, nullable=True)
    final_train_metric: float | None = Field(default=None, nullable=True)
    final_val_loss: float | None = Field(default=None, nullable=True)
    final_val_metric: float | None = Field(default=None, nullable=True)

    # Full epoch-by-epoch curves stored as JSON arrays
    epoch_losses: list | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    epoch_metrics: list | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    epoch_val_losses: list | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    epoch_val_metrics: list | None = Field(default=None, sa_column=Column(JSON, nullable=True))

    # Status: in_progress | success | failed | best
    status: str = Field(default="in_progress", sa_column=Column(String(20), nullable=False))
    error_message: str | None = Field(default=None, nullable=True)

    created_on: datetime | None = Field(default=None, sa_column=Column(DateTime, server_default=func.now()))

    model: Optional["ModelBasic"] = Relationship(back_populates="training_runs")
    evaluation: Optional["ModelEvaluation"] = Relationship(
        back_populates="training_run",
        sa_relationship_kwargs={"uselist": False, "cascade": "all,delete"},
    )


from app.models.evaluation import ModelEvaluation  # noqa: E402
from app.models.ml import ModelBasic  # noqa: E402

ModelTrainingRun.model_rebuild()
