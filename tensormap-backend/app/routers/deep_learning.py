import io
import uuid as uuid_pkg

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse
from sqlmodel import Session

from app.database import get_db
from app.schemas.deep_learning import ModelNameRequest, ModelValidateRequest
from app.services.deep_learning import (
    get_available_model_list,
    get_code_service,
    model_validate_service,
    run_code_service,
)

router = APIRouter(tags=["deep-learning"])


@router.post("/model/validate")
async def validate_model(request: ModelValidateRequest, db: Session = Depends(get_db)):
    body, status_code = model_validate_service(db, incoming=request.model_dump(), project_id=request.project_id)
    return JSONResponse(status_code=status_code, content=body)


@router.post("/model/code")
async def get_code(request: ModelNameRequest, db: Session = Depends(get_db)):
    result, status_code = get_code_service(db, model_name=request.model_name)
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
    body, status_code = run_code_service(db, model_name=request.model_name)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/model/model-list")
async def get_model_list(project_id: uuid_pkg.UUID | None = Query(None), db: Session = Depends(get_db)):
    body, status_code = get_available_model_list(db, project_id=project_id)
    return JSONResponse(status_code=status_code, content=body)
