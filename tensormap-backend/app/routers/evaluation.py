"""GAP-3 and GAP-4: Model evaluation metrics and export endpoints."""

import io

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from app.database import get_db
from app.models.evaluation import ModelEvaluation
from app.models.ml import ModelBasic
from app.services.model_export import export_model_service
from app.shared.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Model Evaluation & Export"])


@router.get("/model/{model_id}/evaluation")
def get_model_evaluation(model_id: int, run_id: int | None = Query(None), db: Session = Depends(get_db)):
    """Return detailed evaluation metrics for a model's training run.

    If run_id is not provided, returns the most recent evaluation.
    """
    model = db.get(ModelBasic, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    query = select(ModelEvaluation).where(ModelEvaluation.model_id == model_id)
    if run_id is not None:
        query = query.where(ModelEvaluation.training_run_id == run_id)
    else:
        query = query.order_by(ModelEvaluation.created_on.desc())

    evaluation = db.exec(query).first()
    if not evaluation:
        raise HTTPException(status_code=404, detail="No evaluation found for this model")

    return {
        "model_id": model_id,
        "model_name": model.model_name,
        "training_run_id": evaluation.training_run_id,
        "overall": {
            "test_loss": evaluation.test_loss,
            "test_metric": evaluation.test_metric,
        },
        "classification": {
            "per_class_metrics": evaluation.per_class_metrics,
            "confusion_matrix": evaluation.confusion_matrix,
            "roc_auc": evaluation.roc_auc,
            "roc_curve_data": evaluation.roc_curve_data,
        },
        "regression": {
            "mae": evaluation.mae,
            "rmse": evaluation.rmse,
            "r_squared": evaluation.r_squared,
        },
        "created_on": evaluation.created_on.isoformat() if evaluation.created_on else None,
    }


@router.get("/model/{model_id}/export")
def export_model(
    model_id: int,
    format: str = Query("h5", description="Export format: h5 or pb"),
    db: Session = Depends(get_db),
):
    """Export trained model weights as h5 (Keras) or pb (SavedModel zip)."""
    file_bytes, filename, media_type, error = export_model_service(db, model_id=model_id, fmt=format)

    if error:
        status = 404 if "not found" in error.lower() else 400
        logger.warning("Export failed for model %d: %s", model_id, error)
        raise HTTPException(status_code=status, detail=error)

    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
