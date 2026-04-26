"""GAP-3: Detailed model evaluation metrics stored after every training run."""

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Column, DateTime, ForeignKey, func
from sqlmodel import Field, Relationship, SQLModel


class ModelEvaluation(SQLModel, table=True):
    """Detailed evaluation metrics computed after a training run."""

    __tablename__ = "model_evaluation"

    id: int | None = Field(default=None, primary_key=True)
    training_run_id: int = Field(
        sa_column=Column(ForeignKey("model_training_run.id", ondelete="CASCADE"), index=True, nullable=False)
    )
    model_id: int = Field(
        sa_column=Column(ForeignKey("model_basic.id", ondelete="CASCADE"), index=True, nullable=False)
    )

    # Overall metrics
    test_loss: float | None = Field(default=None, nullable=True)
    test_metric: float | None = Field(default=None, nullable=True)

    # Classification: per-class precision, recall, F1
    per_class_metrics: dict | None = Field(default=None, sa_column=Column(JSON, nullable=True))

    # Confusion matrix [[TP, FP], [FN, TN]]
    confusion_matrix: list | None = Field(default=None, sa_column=Column(JSON, nullable=True))

    # ROC / AUC (binary classification only)
    roc_auc: float | None = Field(default=None, nullable=True)
    roc_curve_data: dict | None = Field(default=None, sa_column=Column(JSON, nullable=True))

    # Regression metrics
    mae: float | None = Field(default=None, nullable=True)
    rmse: float | None = Field(default=None, nullable=True)
    r_squared: float | None = Field(default=None, nullable=True)

    created_on: datetime | None = Field(default=None, sa_column=Column(DateTime, server_default=func.now()))

    training_run: Optional["ModelTrainingRun"] = Relationship(back_populates="evaluation")


from app.models.training_run import ModelTrainingRun  # noqa: E402

ModelEvaluation.model_rebuild()
