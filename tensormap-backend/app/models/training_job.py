import uuid as uuid_pkg
from datetime import datetime
from enum import StrEnum

from sqlalchemy import JSON, Column, DateTime, ForeignKey
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlmodel import Field, SQLModel


class TrainingStatus(StrEnum):
    """Lifecycle states for a training job.

    A job moves PENDING -> RUNNING -> COMPLETED on the happy path, or to
    FAILED (training raised) / CANCELLED (user requested a stop).
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Store the lowercase ``.value`` (not the member name) so the DB column matches
# the Alembic migration's CHECK constraint and the JSON the API returns.
def _status_values(enum_cls) -> list[str]:
    return [member.value for member in enum_cls]


class TrainingJob(SQLModel, table=True):
    """A single training run for a model, tracked from request to completion.

    Replaces the old stateless fire-and-block flow: each POST /model/run now
    creates a row here so progress, outcome, and cancellation survive the
    request that started them.
    """

    __tablename__ = "training_job"

    id: str = Field(default_factory=lambda: str(uuid_pkg.uuid4()), primary_key=True)
    model_id: int = Field(foreign_key="model_basic.id", index=True)
    # The project this job belongs to. TensorMap has no user/auth model, so jobs
    # are scoped to a project rather than an owner (no per-user authorization).
    project_id: uuid_pkg.UUID | None = Field(
        default=None,
        sa_column=Column(PgUUID(as_uuid=True), ForeignKey("project.id", ondelete="CASCADE"), index=True, nullable=True),
    )
    # native_enum=False stores the value as a portable VARCHAR + CHECK constraint,
    # avoiding a Postgres ENUM type that would need its own migration step.
    status: TrainingStatus = Field(
        default=TrainingStatus.PENDING,
        sa_column=Column(
            SAEnum(TrainingStatus, native_enum=False, length=20, values_callable=_status_values),
            nullable=False,
            index=True,
        ),
    )
    # hyperparams shape: {"optimizer": "adam", "lr": 0.001, "epochs": 50, "batch_size": 32}
    hyperparams: dict | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    started_at: datetime | None = Field(default=None, sa_column=Column(DateTime, nullable=True))
    completed_at: datetime | None = Field(default=None, sa_column=Column(DateTime, nullable=True))
    error_message: str | None = Field(default=None, max_length=2000, nullable=True)
    # Forward-looking columns reserved for later phases (analysis + tuning).
    analysis_cache: dict | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    last_export_download_at: datetime | None = Field(default=None, sa_column=Column(DateTime, nullable=True))
    tuning_session_id: str | None = Field(default=None, max_length=36, nullable=True)
