import uuid as uuid_pkg

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlmodel import Session

from app.database import get_db
from app.schemas.data_process import (
    PreprocessRequest,
    TargetAddRequest,
)
from app.services.data_process import (
    add_target_service,
    delete_one_target_by_id_service,
    get_all_targets_service,
    get_column_stats_service,
    get_correlation_matrix,
    get_data_metrics,
    get_file_data,
    get_one_target_by_id_service,
    preprocess_data,
)
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["data-process"])


@router.post("/data/process/target")
async def add_target(request: TargetAddRequest, db: Session = Depends(get_db)):
    """Set the target field for a dataset file."""
    logger.debug("Adding target field '%s' for file_id=%s", request.target, request.file_id)
    body, status_code = add_target_service(db, file_id=request.file_id, target=request.target)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/data/process/target")
async def get_all_targets(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """List all target field assignments with pagination."""
    body, status_code = get_all_targets_service(db, offset=offset, limit=limit)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/data/process/target/{file_id}")
async def get_target_by_file(file_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
    """Retrieve the target field for a specific file."""
    body, status_code = get_one_target_by_id_service(db, file_id=file_id)
    return JSONResponse(status_code=status_code, content=body)


@router.delete("/data/process/target/{file_id}")
async def delete_target(file_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
    """Remove the target field assignment for a file."""
    logger.debug("Deleting target field for file_id=%s", file_id)
    body, status_code = delete_one_target_by_id_service(db, file_id=file_id)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/data/process/data_metrics/{file_id}")
async def get_metrics(file_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
    """Return descriptive statistics and correlation matrix for a CSV file."""
    body, status_code = get_data_metrics(db, file_id=file_id)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/data/process/stats/{file_id}")
async def get_column_stats(file_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
    """Return per-column descriptive statistics for a CSV file."""
    body, status_code = get_column_stats_service(db, file_id=file_id)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/data/process/correlation/{file_id}")
async def get_correlation(file_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
    """Return the pairwise correlation matrix for all numeric columns in a CSV file."""
    body, status_code = get_correlation_matrix(db, file_id=file_id)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/data/process/file/{file_id}")
async def get_file(file_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
    """Return the full contents of a CSV file as JSON records."""
    body, status_code = get_file_data(db, file_id=file_id)
    return JSONResponse(status_code=status_code, content=body)


@router.post("/data/process/preprocess/{file_id}")
async def preprocess(file_id: uuid_pkg.UUID, request: PreprocessRequest, db: Session = Depends(get_db)):
    """Apply transformations to a CSV file, overwriting it in place."""
    logger.debug("Preprocessing file_id=%s with %d transformations", file_id, len(request.transformations))
    body, status_code = preprocess_data(db, file_id=file_id, transformations=request.transformations)
    return JSONResponse(status_code=status_code, content=body)
