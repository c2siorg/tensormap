import uuid as uuid_pkg

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlmodel import Session

from app.database import get_db
from app.schemas.data_process import (
    ImagePropertiesRequest,
    PreprocessRequest,
    TargetAddRequest,
)
from app.services.data_process import (
    add_target_service,
    delete_one_target_by_id_service,
    get_all_targets_service,
    get_data_metrics,
    get_file_data,
    get_one_target_by_id_service,
    preprocess_data,
    update_image_properties,
)

router = APIRouter(tags=["data-process"])


@router.post("/data/process/target")
async def add_target(request: TargetAddRequest, db: Session = Depends(get_db)):
    body, status_code = add_target_service(db, file_id=request.file_id, target=request.target)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/data/process/target")
async def get_all_targets(db: Session = Depends(get_db)):
    body, status_code = get_all_targets_service(db)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/data/process/target/{file_id}")
async def get_target_by_file(file_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
    body, status_code = get_one_target_by_id_service(db, file_id=file_id)
    return JSONResponse(status_code=status_code, content=body)


@router.delete("/data/process/target/{file_id}")
async def delete_target(file_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
    body, status_code = delete_one_target_by_id_service(db, file_id=file_id)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/data/process/data_metrics/{file_id}")
async def get_metrics(file_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
    body, status_code = get_data_metrics(db, file_id=file_id)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/data/process/file/{file_id}")
async def get_file(file_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
    body, status_code = get_file_data(db, file_id=file_id)
    return JSONResponse(status_code=status_code, content=body)


@router.post("/data/process/preprocess/{file_id}")
async def preprocess(file_id: uuid_pkg.UUID, request: PreprocessRequest, db: Session = Depends(get_db)):
    body, status_code = preprocess_data(db, file_id=file_id, transformations=request.transformations)
    return JSONResponse(status_code=status_code, content=body)


@router.post("/data/process/image_preprocess/{file_id}")
async def image_preprocess(file_id: uuid_pkg.UUID, request: ImagePropertiesRequest, db: Session = Depends(get_db)):
    body, status_code = update_image_properties(
        db,
        file_id=request.fileId,
        image_size=request.image_size,
        batch_size=request.batch_size,
        color_mode=request.color_mode,
        label_mode=request.label_mode,
    )
    return JSONResponse(status_code=status_code, content=body)
