import io
import uuid as uuid_pkg

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse
from sqlmodel import Session

from app.database import get_db
from app.layers.registry import serialize_registry
from app.schemas.deep_learning import ModelNameRequest, ModelSaveRequest, ModelValidateRequest, TrainingConfigRequest
from app.services.deep_learning import (
    check_model_name_service,
    delete_model_service,
    get_available_model_list,
    get_code_service,
    get_model_count_service,
    get_model_graph_service,
    get_training_history_service,
    model_save_service,
    model_validate_service,
    update_training_config_service,
)
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["deep-learning"])


@router.get("/layers")
def get_layers():
    """
    Return the complete layer registry for the frontend.

    This endpoint provides all supported layer types with their specifications,
    enabling the frontend to render a dynamic, categorized sidebar and
    generate appropriate property panels.
    """
    logger.debug("Fetching layer registry")
    return JSONResponse(status_code=200, content=serialize_registry())


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


@router.get("/model/check-name")
def check_model_name(
    model_name: str = Query(),
    project_id: uuid_pkg.UUID | None = Query(None),
    db: Session = Depends(get_db),
):
    """Check if a model name is available."""
    body, status_code = check_model_name_service(db, model_name=model_name, project_id=project_id)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/model/count")
def get_model_count(
    project_id: uuid_pkg.UUID | None = Query(None),
    db: Session = Depends(get_db),
):
    """Get the total count of saved models."""
    body, status_code = get_model_count_service(db, project_id=project_id)
    return JSONResponse(status_code=status_code, content=body)
