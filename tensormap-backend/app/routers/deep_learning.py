import asyncio
import io
import uuid as uuid_pkg

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse
from sqlmodel import Session

from app.database import get_db
from app.schemas.deep_learning import ModelNameRequest, ModelSaveRequest, ModelValidateRequest, TrainingConfigRequest
from app.services.deep_learning import (
    get_available_model_list,
    get_code_service,
    get_model_graph_service,
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
    logger.info("Starting model training: model_name=%s", request.model_name)
    loop = asyncio.get_running_loop()
    body, status_code = await asyncio.to_thread(
        run_code_service, db, model_name=request.model_name, project_id=request.project_id, loop=loop
    )
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


@router.get("/model/model-list")
def get_model_list(
    project_id: uuid_pkg.UUID | None = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Return a paginated list of saved model names, optionally filtered by project."""
    body, status_code = get_available_model_list(db, project_id=project_id, offset=offset, limit=limit)
    return JSONResponse(status_code=status_code, content=body)
