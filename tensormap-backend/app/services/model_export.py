"""Model export service for SavedModel, TFLite, and ONNX formats.

Exports are generated lazily on first request and cached in exports/{job_id}/.
Background cleanup task removes exports older than retention_days.
"""

import re
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlmodel import Session, select

from app.models.training_job import TrainingJob, TrainingStatus
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

EXPORTS_BASE = Path("./exports")


def _sanitize_filename(name: str) -> str:
    """Sanitize model name to prevent path traversal attacks.

    Removes any path separators and ensures the name is safe for filesystem use.
    Only allows alphanumeric, underscore, hyphen, and dot characters.
    """
    # Remove any path separators
    name = name.replace("/", "_").replace("\\", "_")
    # Remove parent directory references
    name = name.replace("..", "_")
    # Only allow safe characters: alphanumeric, underscore, hyphen, dot
    name = re.sub(r"[^\w\-.]", "_", name)
    # Ensure it doesn't start with a dot (hidden file)
    if name.startswith("."):
        name = "_" + name[1:]
    # Limit length
    if len(name) > 100:
        name = name[:100]
    return name


class ONNXUnsupportedError(Exception):
    """Raised when ONNX export is not supported for the model architecture."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__(f"ONNX export not supported: {'; '.join(issues)}")


def export_savedmodel(job_id: str, model_name: str) -> Path:
    """Exports model to SavedModel format, zips it, returns zip path.

    Generates on-demand; returns existing zip if already generated.
    """
    # Sanitize model_name to prevent path traversal
    safe_name = _sanitize_filename(model_name)
    export_dir = EXPORTS_BASE / job_id
    zip_path = export_dir / f"{safe_name}.savedmodel.zip"

    if zip_path.exists():
        logger.info(f"SavedModel export already exists for job {job_id}")
        return zip_path  # Already generated

    model_path = export_dir / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"model.keras not found for job {job_id}")

    # Lazy import to avoid loading TensorFlow in non-training processes
    import tensorflow as tf

    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    savedmodel_dir = export_dir / "savedmodel"
    logger.info(f"Saving SavedModel to {savedmodel_dir}")
    # Keras 3 (bundled with the TensorFlow 2.16.x this project pins) dropped
    # the ``save_format`` argument: ``save_model`` only writes ``.keras``/``.h5``
    # checkpoints, so passing a directory plus ``save_format="tf"`` raised
    #   ValueError: The `save_format` argument is deprecated in Keras 3.
    # and every SavedModel download returned HTTP 500. ``Model.export`` is the
    # supported way to write a SavedModel directory in Keras 3.
    model.export(str(savedmodel_dir))

    # Zip the savedmodel directory
    logger.info(f"Creating zip archive at {zip_path}")
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", str(export_dir), "savedmodel")

    return zip_path


def export_tflite(job_id: str, model_name: str) -> Path:
    """Exports model to TFLite format. Returns .tflite path."""
    # Sanitize model_name to prevent path traversal
    safe_name = _sanitize_filename(model_name)
    export_dir = EXPORTS_BASE / job_id
    tflite_path = export_dir / f"{safe_name}.tflite"

    if tflite_path.exists():
        logger.info(f"TFLite export already exists for job {job_id}")
        return tflite_path

    model_path = export_dir / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"model.keras not found for job {job_id}")

    # Lazy import
    import tensorflow as tf

    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    logger.info("Converting to TFLite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_path.write_bytes(tflite_model)
    logger.info(f"TFLite model saved to {tflite_path}")

    return tflite_path


def validate_onnx_compatible(model, graph_ir: dict | None) -> list[str]:
    """Pre-flight check for tf2onnx compatibility.

    Returns list of issues (empty = compatible).

    Checks:
    - LSTM/GRU with return_sequences=True: require fixed sequence_length in Input spec
    - Unsupported tf.keras ops (tf.while_loop etc.)

    Returns: list[str] of human-readable issue descriptions
    """
    issues = []

    if graph_ir is None:
        # No graph IR available, skip validation
        return issues

    nodes = graph_ir.get("nodes", [])

    # Check for LSTM/GRU with return_sequences=True and variable-length inputs
    has_lstm_gru = False
    has_variable_input = False

    for node in nodes:
        node_params = node.get("node_params", {})
        layer_type = node_params.get("layer_type", "")

        if layer_type in ("lstm", "gru"):
            has_lstm_gru = True
            return_sequences = node_params.get("return_sequences", False)
            if return_sequences:
                # Check if any Input node has None in shape
                for inp_node in nodes:
                    inp_params = inp_node.get("node_params", {})
                    if inp_params.get("layer_type") == "input":
                        shape = inp_params.get("shape", [])
                        if isinstance(shape, (list, tuple)) and None in shape:
                            has_variable_input = True
                            break

    if has_lstm_gru and has_variable_input:
        issues.append(
            "LSTM/GRU with return_sequences=True requires fixed input "
            "sequence length. Set a fixed shape in your Input layer (e.g., [10, 128] instead of [None, 128])."
        )

    return issues


def export_onnx(job_id: str, model_name: str, graph_ir: dict | None = None) -> Path:
    """Exports model to ONNX format using tf2onnx.

    Args:
        job_id: Training job ID
        model_name: Model name for output filename
        graph_ir: Graph IR JSON for compatibility validation

    Returns:
        Path to generated .onnx file

    Raises:
        ONNXUnsupportedError: If model architecture is not ONNX-compatible
        FileNotFoundError: If model.keras doesn't exist
    """
    # Sanitize model_name to prevent path traversal
    safe_name = _sanitize_filename(model_name)
    export_dir = EXPORTS_BASE / job_id
    onnx_path = export_dir / f"{safe_name}.onnx"

    if onnx_path.exists():
        logger.info(f"ONNX export already exists for job {job_id}")
        return onnx_path

    model_path = export_dir / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"model.keras not found for job {job_id}")

    # Lazy import
    import tensorflow as tf

    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Pre-flight check
    issues = validate_onnx_compatible(model, graph_ir)
    if issues:
        logger.warning(f"ONNX export unsupported for job {job_id}: {issues}")
        raise ONNXUnsupportedError(issues)

    # Import tf2onnx
    try:
        import tf2onnx
    except ImportError as e:
        logger.error("tf2onnx not installed")
        raise ONNXUnsupportedError(["tf2onnx library not installed"]) from e

    logger.info("Converting to ONNX")
    # Build input signature
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        # Multiple inputs
        spec = tuple(tf.TensorSpec(shape, tf.float32, name=f"input_{i}") for i, shape in enumerate(input_shape))
    else:
        # Single input
        spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)

    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec)

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    logger.info(f"ONNX model saved to {onnx_path}")
    return onnx_path


def get_export_formats(job_id: str, model_name: str, graph_ir: dict | None = None) -> dict:
    """Returns available export formats with existence status and metadata.

    Returns:
        dict with format info:
        {
            "savedmodel": {
                "available": bool,
                "size_bytes": int | None,
                "expires_at": str | None (ISO format)
            },
            "tflite": {...},
            "onnx": {
                ...,
                "onnx_supported": bool,
                "onnx_issues": list[str] | None
            }
        }
    """
    # Sanitize model_name to prevent path traversal
    safe_name = _sanitize_filename(model_name)
    export_dir = EXPORTS_BASE / job_id
    model_path = export_dir / "model.keras"

    formats = {}

    # Check if model.keras exists
    if not model_path.exists():
        # Training not completed yet
        return {
            "savedmodel": {"available": False, "size_bytes": None, "expires_at": None},
            "tflite": {"available": False, "size_bytes": None, "expires_at": None},
            "onnx": {
                "available": False,
                "size_bytes": None,
                "expires_at": None,
                "onnx_supported": False,
                "onnx_issues": ["Training not completed"],
            },
        }

    # SavedModel
    savedmodel_zip = export_dir / f"{safe_name}.savedmodel.zip"
    formats["savedmodel"] = {
        "available": savedmodel_zip.exists(),
        "size_bytes": savedmodel_zip.stat().st_size if savedmodel_zip.exists() else None,
        "expires_at": _get_expiry_date(savedmodel_zip) if savedmodel_zip.exists() else None,
    }

    # TFLite
    tflite_file = export_dir / f"{safe_name}.tflite"
    formats["tflite"] = {
        "available": tflite_file.exists(),
        "size_bytes": tflite_file.stat().st_size if tflite_file.exists() else None,
        "expires_at": _get_expiry_date(tflite_file) if tflite_file.exists() else None,
    }

    # ONNX - check compatibility without loading the model
    # The validate_onnx_compatible function doesn't actually use the model parameter
    onnx_issues = validate_onnx_compatible(None, graph_ir)
    onnx_supported = len(onnx_issues) == 0

    onnx_file = export_dir / f"{safe_name}.onnx"
    formats["onnx"] = {
        "available": onnx_file.exists(),
        "size_bytes": onnx_file.stat().st_size if onnx_file.exists() else None,
        "expires_at": _get_expiry_date(onnx_file) if onnx_file.exists() else None,
        "onnx_supported": onnx_supported,
        "onnx_issues": onnx_issues if not onnx_supported else None,
    }

    return formats


def _get_expiry_date(file_path: Path) -> str:
    """Calculate expiry date for an export file using configured retention."""
    import os

    retention_days = int(os.getenv("EXPORT_RETENTION_DAYS", "7"))
    mtime = datetime.fromtimestamp(file_path.stat().st_mtime, UTC)
    expires_at = mtime + timedelta(days=retention_days)
    return expires_at.isoformat()


def cleanup_exports(retention_days: int = 7, session: Session = None) -> int:
    """Deletes export directories for jobs completed > retention_days ago.

    Returns count of directories deleted.
    """
    if not EXPORTS_BASE.exists():
        return 0

    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    deleted_count = 0

    # If session provided, query completed jobs
    if session:
        stmt = select(TrainingJob).where(
            TrainingJob.status.in_([TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]),
            TrainingJob.completed_at < cutoff,
        )
        old_jobs = session.exec(stmt).all()

        for job in old_jobs:
            export_dir = EXPORTS_BASE / job.id
            if export_dir.exists():
                try:
                    shutil.rmtree(export_dir)
                    deleted_count += 1
                    logger.info(f"Deleted export directory for job {job.id}")
                except Exception as e:
                    logger.error(f"Failed to delete export directory for job {job.id}: {e}")
    else:
        # Fallback: iterate all directories and check mtime
        for export_dir in EXPORTS_BASE.iterdir():
            if not export_dir.is_dir():
                continue

            # Check directory modification time
            mtime = datetime.fromtimestamp(export_dir.stat().st_mtime, UTC)
            if mtime < cutoff:
                try:
                    shutil.rmtree(export_dir)
                    deleted_count += 1
                    logger.info(f"Deleted export directory {export_dir.name}")
                except Exception as e:
                    logger.error(f"Failed to delete export directory {export_dir.name}: {e}")

    return deleted_count


def delete_job_exports(job_id: str) -> bool:
    """Delete all exports for a specific job.

    Returns True if directory was deleted, False if it didn't exist.
    Raises exception on deletion failure.
    """
    export_dir = EXPORTS_BASE / job_id
    if export_dir.exists():
        shutil.rmtree(export_dir)
        logger.info(f"Deleted export directory for job {job_id}")
        return True
    return False


def delete_model_exports(model_id: int, session: Session) -> int:
    """Delete all exports for all jobs of a given model (best-effort).

    Returns count of directories successfully deleted.
    Logs failures but continues with remaining jobs.
    """
    stmt = select(TrainingJob).where(TrainingJob.model_id == model_id)
    jobs = session.exec(stmt).all()

    deleted_count = 0
    for job in jobs:
        try:
            if delete_job_exports(job.id):
                deleted_count += 1
        except Exception as e:
            logger.error(f"Failed to delete export directory for job {job.id}: {e}")
            # Continue with remaining jobs (best-effort cleanup)
            continue

    return deleted_count
