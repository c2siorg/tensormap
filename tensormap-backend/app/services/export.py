"""Model export service — SavedModel, TFLite, and ONNX formats."""

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import tensorflow as tf
from sqlmodel import Session, select

from app.models.ml import ModelBasic
from app.services.model_run import _helper_generate_json_model_file_location
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

EXPORT_ROOT = "./templates/exports/"


def _ensure_export_dir() -> Path:
    path = Path(EXPORT_ROOT)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_keras_model(model_name: str) -> tf.keras.Model:
    """Load a Keras model from its saved JSON file."""
    json_path = _helper_generate_json_model_file_location(model_name)
    with open(json_path) as f:
        json_string = f.read()
    model = tf.keras.models.model_from_json(json_string)
    return model


def _resp(status_code: int, success: bool, message: str, data=None):
    return {"success": success, "message": message, "data": data}, status_code


def export_savedmodel_service(db: Session, model_name: str) -> tuple:
    """Export model as TensorFlow SavedModel (zipped)."""
    model = db.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()
    if not model:
        return _resp(404, False, "Model not found")

    try:
        keras_model = _load_keras_model(model_name)
    except Exception as e:
        logger.exception("Failed to load model for export: %s", e)
        return _resp(400, False, f"Failed to load model: {e}")

    export_dir = _ensure_export_dir()
    saved_dir = export_dir / f"{model_name}_savedmodel"

    try:
        keras_model.save(str(saved_dir), save_format="tf")
        zip_path = export_dir / f"{model_name}_savedmodel.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in saved_dir.rglob("*"):
                zf.write(file, file.relative_to(saved_dir.parent))
        shutil.rmtree(saved_dir, ignore_errors=True)
        logger.info("SavedModel export complete: %s", zip_path)
        return _resp(
            200,
            True,
            "SavedModel export successful",
            {"file_path": str(zip_path), "file_name": f"{model_name}_savedmodel.zip"},
        )
    except Exception as e:
        logger.exception("SavedModel export failed: %s", e)
        return _resp(500, False, f"Export failed: {e}")


def export_tflite_service(db: Session, model_name: str) -> tuple:
    """Export model as TensorFlow Lite flatbuffer."""
    model = db.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()
    if not model:
        return _resp(404, False, "Model not found")

    try:
        keras_model = _load_keras_model(model_name)
    except Exception as e:
        return _resp(400, False, f"Failed to load model: {e}")

    export_dir = _ensure_export_dir()

    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        tflite_model = converter.convert()
        tflite_path = export_dir / f"{model_name}.tflite"
        tflite_path.write_bytes(tflite_model)
        logger.info("TFLite export complete: %s", tflite_path)
        return _resp(
            200,
            True,
            "TFLite export successful",
            {"file_path": str(tflite_path), "file_name": f"{model_name}.tflite"},
        )
    except Exception as e:
        logger.exception("TFLite export failed: %s", e)
        return _resp(500, False, f"TFLite export failed: {e}")


def export_onnx_service(db: Session, model_name: str) -> tuple:
    """Export model as ONNX (requires tf2onnx)."""
    model = db.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()
    if not model:
        return _resp(404, False, "Model not found")

    try:
        import tf2onnx  # noqa: F401
    except ImportError:
        return _resp(400, False, "tf2onnx is not installed. Run: pip install tf2onnx")

    try:
        keras_model = _load_keras_model(model_name)
    except Exception as e:
        return _resp(400, False, f"Failed to load model: {e}")

    export_dir = _ensure_export_dir()

    try:
        with tempfile.TemporaryDirectory() as tmp:
            saved_path = os.path.join(tmp, "saved")
            keras_model.save(saved_path, save_format="tf")
            onnx_path = export_dir / f"{model_name}.onnx"
            os.system(f"python -m tf2onnx.convert --saved-model {saved_path} --output {onnx_path} --opset 13")
        if not onnx_path.exists():
            return _resp(500, False, "ONNX conversion produced no output file")
        logger.info("ONNX export complete: %s", onnx_path)
        return _resp(
            200,
            True,
            "ONNX export successful",
            {"file_path": str(onnx_path), "file_name": f"{model_name}.onnx"},
        )
    except Exception as e:
        logger.exception("ONNX export failed: %s", e)
        return _resp(500, False, f"ONNX export failed: {e}")
