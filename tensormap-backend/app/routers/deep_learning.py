import asyncio
import io
import json
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
from app.shared.constants import LAYER_REGISTRY_LOCATION
from app.shared.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["deep-learning"])

# MODULE-LEVEL CACHING: Read the registry once when the server boots, not on every request.
try:
    with open(LAYER_REGISTRY_LOCATION) as f:
        _LAYER_REGISTRY = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    logger.critical("Failed to load layer registry: %s", e)
    raise RuntimeError(f"Cannot start server: layer registry unavailable — {e}") from e


@router.post("/model/validate")
async def validate_model(request: ModelValidateRequest, db: Session = Depends(get_db)):
    """Validate a ReactFlow graph as a Keras model and persist the configuration."""
    logger.debug("Validating model for project_id=%s", request.project_id)
    body, status_code = model_validate_service(db, incoming=request.model_dump(), project_id=request.project_id)
    return JSONResponse(status_code=status_code, content=body)


@router.post("/model/save")
async def save_model(request: ModelSaveRequest, db: Session = Depends(get_db)):
    """Save a model architecture from the canvas (no training config)."""
    logger.debug("Saving model architecture: model_name=%s", request.model_name)
    body, status_code = model_save_service(
        db, incoming=request.model.model_dump(), model_name=request.model_name, project_id=request.project_id
    )
    return JSONResponse(status_code=status_code, content=body)


@router.patch("/model/training-config")
async def update_training_config(request: TrainingConfigRequest, db: Session = Depends(get_db)):
    """Set training configuration on a previously saved model."""
    logger.debug("Updating training config for model_name=%s", request.model_name)
    body, status_code = update_training_config_service(
        db, model_name=request.model_name, config=request.model_dump(), project_id=request.project_id
    )
    return JSONResponse(status_code=status_code, content=body)


@router.post("/model/code")
async def get_code(request: ModelNameRequest, db: Session = Depends(get_db)):
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
async def get_model_graph(
    model_name: str,
    project_id: uuid_pkg.UUID | None = Query(None),
    db: Session = Depends(get_db),
):
    """Retrieve the full ReactFlow graph for a saved model."""
    body, status_code = get_model_graph_service(db, model_name=model_name, project_id=project_id)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/model/model-list")
async def get_model_list(
    project_id: uuid_pkg.UUID | None = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Return a paginated list of saved model names, optionally filtered by project."""
    body, status_code = get_available_model_list(db, project_id=project_id, offset=offset, limit=limit)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/layers")
def get_layer_registry():
    """
    Return the data-driven layer registry for dynamic UI generation.
    NOTE: Public endpoint — no auth required by design (layer schema is non-sensitive and needed for UI rendering).
    """
    logger.debug("Fetching unified layer registry")

    if _LAYER_REGISTRY is None:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Unified layer registry is missing or corrupted on the server."},
        )

    from fastapi import HTTPException  # Ensure this is imported at the top

    if _LAYER_REGISTRY is None:
        raise HTTPException(
            status_code=500,
            detail={"success": False, "message": "Unified layer registry is missing or corrupted on the server."},
        )
