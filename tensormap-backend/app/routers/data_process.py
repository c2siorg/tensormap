import uuid as uuid_pkg
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from sqlmodel import Session

from app.config import get_settings
from app.database import get_db
from app.models import DataFile
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
    get_dataset_backup_status_service,
    get_file_data,
    get_one_target_by_id_service,
    preprocess_data,
    revert_data_process_service,
)
from app.services.dataset_quality import analyze_dataset_quality
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["data-process"])


@router.get("/data/process/quality/{file_id}")
def get_dataset_quality(
    file_id: uuid_pkg.UUID,
    target_field: str | None = Query(None),
    problem_type_id: int | None = Query(None, ge=1, le=3),
    training_split: float | None = Query(None),
    db: Session = Depends(get_db),
):
    """Return a read-only quality report for a dataset and its selected target."""
    body, status_code = analyze_dataset_quality(
        db,
        file_id=file_id,
        target_field=target_field,
        problem_type_id=problem_type_id,
        training_split=training_split,
    )
    return JSONResponse(status_code=status_code, content=body)


@router.post("/data/process/target")
def add_target(request: TargetAddRequest, db: Session = Depends(get_db)):
    """Set the target field for a dataset file."""
    logger.debug("Adding target field '%s' for file_id=%s", request.target, request.file_id)
    body, status_code = add_target_service(db, file_id=request.file_id, target=request.target)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/data/process/target")
def get_all_targets(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """List all target field assignments with pagination."""
    body, status_code = get_all_targets_service(db, offset=offset, limit=limit)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/data/process/target/{file_id}")
def get_target_by_file(file_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
    """Retrieve the target field for a specific file."""
    body, status_code = get_one_target_by_id_service(db, file_id=file_id)
    return JSONResponse(status_code=status_code, content=body)


@router.delete("/data/process/target/{file_id}")
def delete_target(file_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
    """Remove the target field assignment for a file."""
    logger.debug("Deleting target field for file_id=%s", file_id)
    body, status_code = delete_one_target_by_id_service(db, file_id=file_id)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/data/process/data_metrics/{file_id}")
def get_metrics(file_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
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
async def get_file(
    file_id: uuid_pkg.UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """Return a paginated slice of a CSV file's contents with pagination metadata."""
    body, status_code = get_file_data(db, file_id=file_id, page=page, page_size=page_size)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/data/process/download/{file_id}", response_class=FileResponse)
def download_dataset(file_id: uuid_pkg.UUID, db: Session = Depends(get_db)) -> FileResponse:
    """Download a CSV dataset exactly as stored, without modifying it."""
    data_file = db.get(DataFile, file_id)
    if data_file is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if data_file.file_type != "csv":
        raise HTTPException(status_code=400, detail="Only CSV datasets can be downloaded as CSV")

    upload_folder = Path(get_settings().upload_folder).resolve()
    file_path = (upload_folder / data_file.disk_name).resolve()
    if not file_path.is_relative_to(upload_folder):
        logger.warning("Dataset download path outside upload folder for file_id=%s", file_id)
        raise HTTPException(status_code=404, detail="Dataset file not found")

    try:
        with file_path.open("rb"):
            pass
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Dataset file not found") from error
    except OSError as error:
        logger.exception("Dataset file unavailable for download, file_id=%s", file_id)
        raise HTTPException(status_code=500, detail="Dataset file is unavailable for download") from error

    filename = Path(data_file.file_name.replace("\\", "/")).name or "dataset"
    if Path(filename).suffix.lower() != ".csv":
        filename += ".csv"

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="text/csv",
    )


@router.post("/data/process/preprocess/{file_id}")
def preprocess(file_id: uuid_pkg.UUID, request: PreprocessRequest, db: Session = Depends(get_db)):
    """Apply transformations to a CSV file, overwriting it in place."""
    logger.debug("Preprocessing file_id=%s with %d transformations", file_id, len(request.transformations))
    body, status_code = preprocess_data(db, file_id=file_id, transformations=request.transformations)
    return JSONResponse(status_code=status_code, content=body)


@router.post("/data/process/revert/{file_id}")
def revert_preprocess(file_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
    """Revert a CSV file to its pre-processed state using the backup file."""
    logger.debug("Reverting preprocessing for file_id=%s", file_id)
    body, status_code = revert_data_process_service(db, file_id=file_id)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/data/process/backup/{file_id}")
def get_backup_status(file_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
    """Check if a preprocessing backup exists for a CSV file and return its details."""
    body, status_code = get_dataset_backup_status_service(db, file_id=file_id)
    return JSONResponse(status_code=status_code, content=body)
