"""API endpoints for holdout model evaluation, error diagnostics, and batch prediction."""

from fastapi import APIRouter, Depends, Response
from fastapi.concurrency import run_in_threadpool
from sqlmodel import Session

from app.database import get_db
from app.schemas.evaluation import (
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from app.services.evaluation_service import EvaluationService
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/model/analysis", tags=["evaluation"])

_service = EvaluationService()


@router.post(
    "/{job_id}/evaluate",
    response_model=BatchEvaluationResponse,
    summary="Evaluate trained model on holdout dataset",
)
async def evaluate_holdout_dataset(
    job_id: str,
    payload: BatchEvaluationRequest,
    session: Session = Depends(get_db),
) -> BatchEvaluationResponse:
    """Evaluates a trained model against an uploaded test or validation dataset.

    Computes comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, Log Loss
    for classification; MSE, RMSE, MAE, R², MAPE, Max Error for regression) and
    identifies top 10 misclassified or largest residual error samples.
    """
    logger.info("Evaluating job %s on dataset %s", job_id, payload.file_id)
    return await run_in_threadpool(
        _service.evaluate_holdout_dataset,
        job_id=job_id,
        file_id=payload.file_id,
        target_field=payload.target_field,
        batch_size=payload.batch_size,
        session=session,
    )


@router.post(
    "/{job_id}/batch-predict",
    response_model=BatchPredictionResponse,
    summary="Generate batch predictions on dataset",
)
async def run_batch_prediction(
    job_id: str,
    payload: BatchPredictionRequest,
    session: Session = Depends(get_db),
) -> BatchPredictionResponse:
    """Generates batch predictions on an uploaded dataset and returns preview rows."""
    logger.info("Running batch predictions for job %s on dataset %s", job_id, payload.file_id)
    _, response = await run_in_threadpool(
        _service.run_batch_prediction,
        job_id=job_id,
        file_id=payload.file_id,
        batch_size=payload.batch_size,
        session=session,
    )
    return response


@router.post(
    "/{job_id}/batch-predict/export",
    summary="Download batch predictions as CSV",
)
async def export_batch_predictions(
    job_id: str,
    payload: BatchPredictionRequest,
    session: Session = Depends(get_db),
) -> Response:
    """Runs batch predictions and streams the full scored dataset as a downloadable CSV."""
    logger.info("Exporting batch prediction CSV for job %s on dataset %s", job_id, payload.file_id)
    csv_content = await run_in_threadpool(
        _service.export_batch_predictions_csv,
        job_id=job_id,
        file_id=payload.file_id,
        batch_size=payload.batch_size,
        session=session,
    )
    filename = f"predictions_{job_id[:8]}.csv"
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )
