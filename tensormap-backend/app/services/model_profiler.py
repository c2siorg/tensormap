"""Pre-training model architecture complexity, parameter profiling, and memory estimation service."""

import math
import os
from typing import Any

from sqlmodel import Session, select

from app.models.ml import ModelBasic
from app.schemas.model_profiler import (
    ArchitecturalWarning,
    LayerProfileDetail,
    ModelComplexityProfile,
)
from app.shared.constants import MODEL_GENERATION_LOCATION, MODEL_GENERATION_TYPE
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

_tf = None


def _get_tensorflow():
    """Import TensorFlow lazily so app startup does not hard-fail on ML deps."""
    global _tf
    if _tf is not None:
        return _tf
    try:
        import tensorflow as tf

        _tf = tf
        return _tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow import failed. Install compatible tensorflow/protobuf versions for this runtime."
        ) from exc


def _validate_model_path(model_name: str) -> str:
    """Resolve model JSON path and guard against directory traversal."""
    path = os.path.realpath(os.path.join(MODEL_GENERATION_LOCATION, model_name + MODEL_GENERATION_TYPE))
    base_dir = os.path.realpath(MODEL_GENERATION_LOCATION)
    if not path.startswith(base_dir + os.sep) and path != base_dir:
        raise ValueError("Invalid model path: escapes model directory")
    return path


def _count_weight_params(weights: list[Any]) -> int:
    """Calculates total scalar elements across a list of weight tensors or variables."""
    return sum(int(math.prod(w.shape)) for w in weights if getattr(w, "shape", None) is not None)


def profile_keras_model(model: Any, batch_size: int = 32) -> ModelComplexityProfile:
    """Profiles a built Keras model architecture for parameter distribution and memory usage."""
    if not getattr(model, "built", False):
        in_shape = getattr(model, "input_shape", None)
        if in_shape is not None:
            try:
                model.build(in_shape)
            except Exception as e:
                logger.debug("Could not auto-build model with shape %s: %s", in_shape, e)

    layer_details: list[LayerProfileDetail] = []
    total_activation_bytes = 0
    current_shape = getattr(model, "input_shape", None)

    for layer in model.layers:
        trainable = _count_weight_params(layer.trainable_weights)
        non_trainable = _count_weight_params(layer.non_trainable_weights)
        total_p = trainable + non_trainable
        out_shape_list: list[Any] | None = None
        try:
            raw_shape = getattr(layer, "output_shape", None)
            if raw_shape is None and current_shape is not None:
                try:
                    raw_shape = layer.compute_output_shape(current_shape)
                except Exception:
                    raw_shape = None
            if raw_shape is not None:
                current_shape = raw_shape
            if isinstance(raw_shape, tuple):
                out_shape_list = list(raw_shape)
                layer_elems = 1
                for dim in raw_shape:
                    layer_elems *= max(1, batch_size if dim is None else int(dim))
                total_activation_bytes += layer_elems * 4
            elif isinstance(raw_shape, list):
                out_shape_list = [list(s) if isinstance(s, tuple) else str(s) for s in raw_shape]
        except Exception as e:
            logger.debug("Could not inspect output_shape for %s: %s", layer.name, e)

        layer_details.append(
            LayerProfileDetail(
                layer_name=layer.name,
                layer_type=layer.__class__.__name__,
                trainable_params=trainable,
                non_trainable_params=non_trainable,
                total_params=total_p,
                param_memory_kb=round(total_p * 4.0 / 1024.0, 2),
                output_shape=out_shape_list,
            )
        )

    tot_trainable = sum(d.trainable_params for d in layer_details)
    tot_non_trainable = sum(d.non_trainable_params for d in layer_details)
    tot_params = tot_trainable + tot_non_trainable
    warnings: list[ArchitecturalWarning] = []

    if tot_params == 0:
        warnings.append(
            ArchitecturalWarning(category="no_parameters", layer_name="model", message="Model has 0 parameters.")
        )
    elif tot_trainable == 0:
        warnings.append(
            ArchitecturalWarning(category="zero_trainable", layer_name="model", message="All parameters are frozen.")
        )

    if tot_params > 1000 and len(layer_details) >= 3:
        for d in layer_details:
            if (d.total_params / tot_params) >= 0.85:
                warnings.append(
                    ArchitecturalWarning(
                        category="parameter_concentration",
                        layer_name=d.layer_name,
                        message=f"Layer '{d.layer_name}' accounts for >=85% of total parameters.",
                    )
                )

    # Detect extreme bottlenecks on hidden/intermediate representations, excluding the output layer
    num_layers = len(layer_details)
    for i in range(num_layers - 2):
        curr, nxt = layer_details[i], layer_details[i + 1]
        if curr.output_shape and nxt.output_shape:
            c_dim, n_dim = curr.output_shape[-1], nxt.output_shape[-1]
            if isinstance(c_dim, int) and isinstance(n_dim, int) and c_dim >= 64 and n_dim <= max(1, c_dim // 8):
                warnings.append(
                    ArchitecturalWarning(
                        category="extreme_bottleneck",
                        layer_name=nxt.layer_name,
                        message=(
                            f"Steep compression from '{curr.layer_name}' ({c_dim}) to '{nxt.layer_name}' ({n_dim})."
                        ),
                    )
                )

    return ModelComplexityProfile(
        total_layers=len(layer_details),
        total_params=tot_params,
        trainable_params=tot_trainable,
        non_trainable_params=tot_non_trainable,
        param_memory_mb=round(tot_params * 4.0 / (1024.0**2), 4),
        estimated_activation_memory_mb=round(total_activation_bytes / (1024.0**2), 4),
        layers=layer_details,
        warnings=warnings,
    )


def profile_model_service(db: Session, model_id: int, batch_size: int = 32) -> tuple[dict[str, Any], int]:
    """Retrieves a saved model, instantiates its architecture, and calculates complexity profile."""
    model_record = db.exec(select(ModelBasic).where(ModelBasic.id == model_id)).first()
    if not model_record:
        return {"success": False, "message": "Model not found", "data": None}, 404

    model_name = model_record.model_name
    try:
        model_path = _validate_model_path(model_name)
    except ValueError as e:
        logger.warning("Invalid model path for '%s': %s", model_name, e)
        return {"success": False, "message": "Invalid model path", "data": None}, 400

    if not os.path.exists(model_path):
        return {"success": False, "message": f"Model architecture file not found: '{model_name}'", "data": None}, 404

    try:
        with open(model_path, encoding="utf-8") as f:
            json_string = f.read()

        tf_module = _get_tensorflow()
        k_model = tf_module.keras.models.model_from_json(json_string)

        profile = profile_keras_model(k_model, batch_size=batch_size)
        return {
            "success": True,
            "message": "Model architecture profiled successfully",
            "data": profile.model_dump(),
        }, 200
    except Exception as e:
        logger.exception("Failed to profile model '%s': %s", model_name, e)
        return {"success": False, "message": "Failed to profile model architecture.", "data": None}, 500
