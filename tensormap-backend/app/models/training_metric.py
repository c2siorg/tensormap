from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, Index
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    """Timezone-aware UTC now (datetime.utcnow is deprecated in 3.12)."""
    return datetime.now(UTC)


class TrainingMetric(SQLModel, table=True):
    """One metric value recorded for a single epoch of a training job.

    Metrics are stored long-form (one row per metric per epoch) so new metric
    names can be added without a schema change. Charts read them back grouped
    by epoch via the composite index below.
    """

    __tablename__ = "training_metric"

    id: int | None = Field(default=None, primary_key=True)
    job_id: str = Field(foreign_key="training_job.id", index=True)
    epoch: int
    metric_name: str = Field(max_length=50)  # "loss", "accuracy", "val_loss", "val_accuracy"
    metric_value: float
    recorded_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime, nullable=False))

    __table_args__ = (Index("ix_metric_job_epoch", "job_id", "epoch"),)
