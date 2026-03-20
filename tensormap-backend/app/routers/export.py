"""Export router — download trained models in various formats."""

import os

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, JSONResponse
from sqlmodel import Session

from app.database import get_db
from app.services.export import (
    export_onnx_service,
    export_savedmodel_service,
    export_tflite_service,
)
from app.shared.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["export"])


def _file_response(result: dict, status_code: int):
    if status_code != 200:
        return JSONResponse(status_code=status_code, content=result)
    file_path = result["data"]["file_path"]
    file_name = result["data"]["file_name"]
    if not os.path.exists(file_path):
        return JSONResponse(status_code=500, content={"success": False, "message": "Export file not found"})
    media = "application/octet-stream"
    if file_name.endswith(".zip"):
        media = "application/zip"
    elif file_name.endswith(".tflite") or file_name.endswith(".onnx"):
        media = "application/octet-stream"
    return FileResponse(path=file_path, media_type=media, filename=file_name)


@router.get("/model/{model_name}/export/savedmodel")
def export_savedmodel(model_name: str, db: Session = Depends(get_db)):
    """Download model as TensorFlow SavedModel (zip)."""
    result, status_code = export_savedmodel_service(db, model_name)
    return _file_response(result, status_code)


@router.get("/model/{model_name}/export/tflite")
def export_tflite(model_name: str, db: Session = Depends(get_db)):
    """Download model as TensorFlow Lite (.tflite)."""
    result, status_code = export_tflite_service(db, model_name)
    return _file_response(result, status_code)


@router.get("/model/{model_name}/export/onnx")
def export_onnx(model_name: str, db: Session = Depends(get_db)):
    """Download model as ONNX (.onnx)."""
    result, status_code = export_onnx_service(db, model_name)
    return _file_response(result, status_code)
