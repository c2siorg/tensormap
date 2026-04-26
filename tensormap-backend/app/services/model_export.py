"""GAP-4: Export trained Keras model weights in h5 or SavedModel format."""

import os
import tempfile
import zipfile

import tensorflow as tf
from sqlmodel import Session, select

from app.models.ml import ModelBasic
from app.shared.constants import MODEL_GENERATION_LOCATION, MODEL_GENERATION_TYPE
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

# Maximum export file size: 500MB
MAX_EXPORT_SIZE_MB = 500


def _load_compiled_model(model_configs: ModelBasic) -> tf.keras.Model:
    """Load and compile a Keras model from its saved JSON file."""
    json_path = os.path.realpath(
        os.path.join(MODEL_GENERATION_LOCATION, model_configs.model_name + MODEL_GENERATION_TYPE)
    )
    base_dir = os.path.realpath(MODEL_GENERATION_LOCATION)
    if not json_path.startswith(base_dir + os.sep):
        raise ValueError("Invalid model path")

    with open(json_path) as f:
        json_string = f.read()

    model = tf.keras.models.model_from_json(json_string)

    if model_configs.loss == "sparse_categorical_crossentropy":
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        loss = tf.keras.losses.MeanSquaredError()

    model.compile(
        optimizer=model_configs.optimizer or "adam",
        loss=loss,
        metrics=[model_configs.metric or "accuracy"],
    )
    return model


def export_model_service(db: Session, model_id: int, fmt: str) -> tuple:
    """Export a model in the requested format.

    Returns:
        Tuple of (file_bytes, filename, media_type, error_message)
        On success: (bytes, str, str, None)
        On error: (None, None, None, str)
    """
    model_configs = db.exec(select(ModelBasic).where(ModelBasic.id == model_id)).first()
    if not model_configs:
        return None, None, None, "Model not found"

    if fmt not in ("h5", "pb"):
        return None, None, None, "Unsupported format. Use 'h5' or 'pb'."

    try:
        model = _load_compiled_model(model_configs)
    except Exception as exc:
        logger.exception("Failed to load model %s for export", model_id)
        return None, None, None, str(exc)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            if fmt == "h5":
                out_path = os.path.join(tmpdir, f"{model_configs.model_name}.h5")
                model.save(out_path, save_format="h5")
                with open(out_path, "rb") as f:
                    file_bytes = f.read()
                return file_bytes, f"{model_configs.model_name}.h5", "application/octet-stream", None

            else:  # pb / SavedModel
                saved_model_dir = os.path.join(tmpdir, model_configs.model_name)
                model.save(saved_model_dir, save_format="tf")

                # Zip the SavedModel directory
                zip_path = os.path.join(tmpdir, f"{model_configs.model_name}_savedmodel.zip")
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for root, _, files in os.walk(saved_model_dir):
                        for file in files:
                            abs_path = os.path.join(root, file)
                            arcname = os.path.relpath(abs_path, tmpdir)
                            zf.write(abs_path, arcname)

                with open(zip_path, "rb") as f:
                    file_bytes = f.read()
                return file_bytes, f"{model_configs.model_name}_savedmodel.zip", "application/zip", None

    except Exception as exc:
        logger.exception("Export failed for model %s", model_id)
        return None, None, None, str(exc)
