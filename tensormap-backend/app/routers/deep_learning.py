import asyncio
import io
import uuid as uuid_pkg

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse
from sqlmodel import Session

from app.database import get_db
from app.schemas.deep_learning import ModelNameRequest, ModelSaveRequest, ModelValidateRequest, TrainingConfigRequest
from app.services.deep_learning import (
    delete_model_service,
    get_available_model_list,
    get_code_service,
    get_model_graph_service,
    get_training_history_service,
    model_save_service,
    model_validate_service,
    run_code_service,
    update_training_config_service,
)
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["deep-learning"])


@router.post("/model/validate")
def validate_model(request: ModelValidateRequest, db: Session = Depends(get_db)):
    """Validate a ReactFlow graph as a Keras model and persist the configuration."""
    logger.debug("Validating model for project_id=%s", request.project_id)
    body, status_code = model_validate_service(db, incoming=request.model_dump(), project_id=request.project_id)
    return JSONResponse(status_code=status_code, content=body)


@router.post("/model/save")
def save_model(request: ModelSaveRequest, db: Session = Depends(get_db)):
    """Save a model architecture from the canvas (no training config)."""
    logger.debug("Saving model architecture: model_name=%s", request.model_name)
    body, status_code = model_save_service(
        db, incoming=request.model.model_dump(), model_name=request.model_name, project_id=request.project_id
    )
    return JSONResponse(status_code=status_code, content=body)


@router.patch("/model/training-config")
def update_training_config(request: TrainingConfigRequest, db: Session = Depends(get_db)):
    """Set training configuration on a previously saved model."""
    logger.debug("Updating training config for model_name=%s", request.model_name)
    body, status_code = update_training_config_service(
        db, model_name=request.model_name, config=request.model_dump(), project_id=request.project_id
    )
    return JSONResponse(status_code=status_code, content=body)


@router.post("/model/code")
def get_code(request: ModelNameRequest, db: Session = Depends(get_db)):
    """Generate and download a Python training script for a saved model."""
    logger.debug("Generating code for model_name=%s", request.model_name)
    result, status_code = get_code_service(db, model_name=request.model_name, project_id=request.project_id)
    if status_code == 200:
        temp_file = io.BytesIO(result["content"].encode())
        return StreamingResponse(
            temp_file,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={result['file_name']}"},
        )
    return JSONResponse(status_code=status_code, content=result)


@router.post("/model/run")
async def run_model(request: ModelNameRequest, db: Session = Depends(get_db)):
    """Train a saved model in a background thread and stream progress via Socket.IO."""

    request_id = str(uuid_pkg.uuid4())[:8]
    logger.info(
        "[%s] Starting model training: model_name=%s project_id=%s", request_id, request.model_name, request.project_id
    )
    loop = asyncio.get_running_loop()
    try:
        body, status_code = await asyncio.to_thread(
            run_code_service, db, model_name=request.model_name, project_id=request.project_id, loop=loop
        )
    except Exception:
        logger.exception("[%s] Unhandled exception during model training", request_id)
        return JSONResponse(
            status_code=500,
            content={"error": "Model training encountered an unexpected error.", "request_id": request_id},
        )
    if status_code >= 400:
        logger.warning("[%s] Training returned error %d: %s", request_id, status_code, body)
    else:
        logger.info("[%s] Training completed successfully", request_id)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/model/{model_name}/graph")
def get_model_graph(
    model_name: str,
    project_id: uuid_pkg.UUID | None = Query(None),
    db: Session = Depends(get_db),
):
    """Retrieve the full ReactFlow graph for a saved model."""
    body, status_code = get_model_graph_service(db, model_name=model_name, project_id=project_id)
    return JSONResponse(status_code=status_code, content=body)


@router.delete("/model/{model_id}")
async def delete_model(
    model_id: int,
    db: Session = Depends(get_db),
):
    """Delete a saved model and its associated configuration records."""
    logger.info("Deleting model id=%s", model_id)
    body, status_code = delete_model_service(db, model_id=model_id)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/model/model-list")
def get_model_list(
    project_id: uuid_pkg.UUID | None = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Return a paginated list of saved model names (backward compatible)."""
    body, status_code = get_available_model_list(db, project_id=project_id, offset=offset, limit=limit)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/model/training-history")
def get_training_history(
    project_id: uuid_pkg.UUID | None = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Return a paginated list of models with enriched metadata for training history view."""
    body, status_code = get_training_history_service(db, project_id=project_id, offset=offset, limit=limit)
    return JSONResponse(status_code=status_code, content=body)
