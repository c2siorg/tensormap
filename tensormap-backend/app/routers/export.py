"""Export router for trained model downloads (SavedModel, TFLite, ONNX)."""

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlmodel import Session

from app.database import get_db, get_session
from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus
from app.services.model_export import (
    ONNXUnsupportedError,
    export_onnx,
    export_savedmodel,
    export_tflite,
    get_export_formats,
)
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/model/export", tags=["export"])


def _verify_job_completed(job_id: str, session: Session) -> TrainingJob:
    """Verify that a training job exists and is completed.

    Args:
        job_id: Training job ID
        session: Database session

    Returns:
        TrainingJob instance

    Raises:
        HTTPException: 404 if job not found, 400 if job not completed
    """
    job = session.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    if job.status != TrainingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Export only available for completed training jobs. Current status: {job.status}",
        )

    return job


@router.get("/{job_id}/formats")
async def list_export_formats(
    job_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """Returns available export formats with metadata.

    Response:
    {
        "formats": {
            "savedmodel": {
                "available": bool,
                "size_bytes": int | null,
                "expires_at": str | null
            },
            "tflite": {...},
            "onnx": {
                ...,
                "onnx_supported": bool,
                "onnx_issues": list[str] | null
            }
        }
    }
    """
    # Check if job exists and is completed
    job = _verify_job_completed(job_id, db)

    # Get model to retrieve name and graph_ir
    model = db.get(ModelBasic, job.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    formats = get_export_formats(job_id, model.model_name, model.graph_ir)

    return {"formats": formats}


@router.get("/{job_id}")
async def download_export(
    job_id: str,
    format: Literal["savedmodel", "tflite", "onnx"],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> FileResponse:
    """Download exported model in specified format.

    Query params:
        format: savedmodel | tflite | onnx

    Returns:
        FileResponse with appropriate content-type and filename

    Updates last_export_download_at on training_job after download.
    Runs export generation in threadpool to avoid blocking the event loop.
    """
    # Check if job exists and is completed
    job = _verify_job_completed(job_id, db)

    # Get model to retrieve name
    model = db.get(ModelBasic, job.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    model_name = model.model_name

    # Generate export based on format (run in threadpool to avoid blocking)
    try:
        if format == "savedmodel":
            file_path = await run_in_threadpool(export_savedmodel, job_id, model_name)
            filename = f"{model_name}.savedmodel.zip"
            media_type = "application/zip"

        elif format == "tflite":
            file_path = await run_in_threadpool(export_tflite, job_id, model_name)
            filename = f"{model_name}.tflite"
            media_type = "application/octet-stream"

        elif format == "onnx":
            file_path = await run_in_threadpool(export_onnx, job_id, model_name, model.graph_ir)
            filename = f"{model_name}.onnx"
            media_type = "application/octet-stream"

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

    except FileNotFoundError as e:
        logger.error(f"Export file not found for job {job_id}: {e}")
        raise HTTPException(
            status_code=404,
            detail="Model file not found. Training may not be completed yet.",
        ) from e

    except ONNXUnsupportedError as e:
        logger.warning(f"ONNX export unsupported for job {job_id}: {e.issues}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "onnx_unsupported",
                "issues": e.issues,
                "suggestion": "Use SavedModel or TFLite format instead",
            },
        ) from e

    except Exception as e:
        logger.error(f"Export failed for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}") from e

    # Update last_export_download_at in background using a new session
    def update_download_timestamp():
        with get_session() as session:
            job_to_update = session.get(TrainingJob, job_id)
            if job_to_update:
                job_to_update.last_export_download_at = datetime.now(UTC)
                session.add(job_to_update)
                session.commit()

    background_tasks.add_task(update_download_timestamp)

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type,
    )
