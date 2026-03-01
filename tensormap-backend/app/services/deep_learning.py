import contextlib
import json
import os
import re
import uuid as uuid_pkg
from typing import Any

import tensorflow as tf
from flatten_json import flatten
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

import app.shared.errors as errors
from app.models import ModelBasic, ModelConfigs
from app.services.code_generation import generate_code
from app.services.model_generation import model_generation
from app.services.model_run import model_run
from app.shared.constants import (
    CODE,
    DATASET,
    DL_MODEL,
    FILE_ID,
    FILE_TARGET,
    MODEL_EPOCHS,
    MODEL_GENERATION_LOCATION,
    MODEL_GENERATION_TYPE,
    MODEL_METRIC,
    MODEL_NAME,
    MODEL_OPTIMIZER,
    MODEL_TRAINING_SPLIT,
    PROBLEM_TYPE,
)
from app.shared.enums import ProblemType
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def _resp(status_code: int, success: bool, message: str, data: Any = None) -> tuple:
    """Build a standard API response tuple of (body_dict, status_code)."""
    return {"success": success, "message": message, "data": data}, status_code


def _sanitize_model_name(name: str) -> str:
    """Validate a model name for safe use in file paths."""
    name = name.strip()
    if not name or not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
        raise ValueError(f"Invalid model name: {name!r}. Only alphanumeric, hyphens, and underscores are allowed.")
    return name


def _build_model_summary(keras_model) -> dict:
    """Extract a structured architecture summary from a built Keras model."""
    layers = []
    for layer in keras_model.layers:
        try:
            output_shape = str(layer.output_shape)
        except AttributeError:
            output_shape = "unknown"
        layers.append(
            {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": output_shape,
                "param_count": int(layer.count_params()),
            }
        )

    total = int(keras_model.count_params())
    trainable = int(sum(w.numpy().size for w in keras_model.trainable_weights))
    return {
        "layers": layers,
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": total - trainable,
    }


def model_validate_service(db: Session, incoming: dict, project_id: uuid_pkg.UUID | None = None) -> tuple:
    """Validate a model graph with Keras, persist the configuration, and save the JSON file."""
    model_generated = model_generation(model_params=incoming["model"])

    try:
        keras_model = tf.keras.models.model_from_json(json.dumps(model_generated))
    except Exception as e:
        logger.error("Model validation error: %s", str(e))
        for error in errors.err_msgs:
            if error in str(e):
                return _resp(400, False, errors.err_msgs[error])
        return _resp(400, False, "Model validation failed. Please recheck the model and inputs")

    code = incoming[CODE]
    try:
        code[DL_MODEL][MODEL_NAME] = _sanitize_model_name(code[DL_MODEL][MODEL_NAME])
    except ValueError as e:
        return _resp(400, False, str(e))

    existing = db.exec(select(ModelBasic).where(ModelBasic.model_name == code[DL_MODEL][MODEL_NAME])).first()
    if existing:
        return _resp(400, False, "Model name already used. Use a different name")

    if code[PROBLEM_TYPE] in (ProblemType.CLASSIFICATION, ProblemType.IMAGE_CLASSIFICATION):
        loss = "sparse_categorical_crossentropy"
    else:
        loss = "mean_squared_error"

    model = ModelBasic(
        model_name=code[DL_MODEL][MODEL_NAME],
        file_id=code[DATASET][FILE_ID],
        project_id=project_id,
        model_type=code[PROBLEM_TYPE],
        target_field=code[DATASET].get(FILE_TARGET),
        training_split=code[DATASET][MODEL_TRAINING_SPLIT],
        optimizer=code[DL_MODEL][MODEL_OPTIMIZER],
        metric=code[DL_MODEL][MODEL_METRIC],
        epochs=code[DL_MODEL][MODEL_EPOCHS],
        loss=loss,
    )

    configs = []
    params = flatten(incoming, separator=".")
    for param in params:
        if params[param] is not None:
            configs.append(ModelConfigs(parameter=param, value=str(params[param])))

    try:
        db.add(model)
        db.flush()
        db.refresh(model)
        for cfg in configs:
            cfg.model_id = model.id
        db.add_all(configs)
        db.flush()
    except IntegrityError as e:
        db.rollback()
        logger.exception("IntegrityError saving model: %s", str(e))
        for error in errors.err_msgs:
            if error in str(e):
                return _resp(400, False, errors.err_msgs[error])
        return _resp(400, False, "Model saving failed. Please recheck the model configs")

    try:
        model_path = os.path.join(MODEL_GENERATION_LOCATION, code[DL_MODEL][MODEL_NAME] + MODEL_GENERATION_TYPE)
        with open(model_path, "w+") as f:
            f.write(json.dumps(model_generated) + "\n")
        db.commit()
    except OSError:
        db.rollback()
        logger.exception("Error writing model file")
        return _resp(400, False, "Model validated but failed to save")

    logger.info("Model '%s' validated and saved successfully", code[DL_MODEL][MODEL_NAME])
    return _resp(200, True, "Model Validation and saving successful", {"summary": _build_model_summary(keras_model)})


def model_save_service(db: Session, incoming: dict, model_name: str, project_id: uuid_pkg.UUID | None = None) -> tuple:
    """Validate a model graph with Keras and save architecture only (no training config)."""
    model_generated = model_generation(model_params=incoming)

    try:
        keras_model = tf.keras.models.model_from_json(json.dumps(model_generated))
    except Exception as e:
        logger.error("Model validation error: %s", str(e))
        for error in errors.err_msgs:
            if error in str(e):
                return _resp(400, False, errors.err_msgs[error])
        return _resp(400, False, "Model validation failed. Please recheck the model and inputs")

    try:
        model_name = _sanitize_model_name(model_name)
    except ValueError as e:
        return _resp(400, False, str(e))

    existing = db.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()
    if existing:
        return _resp(400, False, "Model name already used. Use a different name")

    model = ModelBasic(
        model_name=model_name,
        project_id=project_id,
    )

    configs = []
    params = flatten(incoming, separator=".")
    for param in params:
        if params[param] is not None:
            configs.append(ModelConfigs(parameter=param, value=str(params[param])))

    try:
        db.add(model)
        db.flush()
        db.refresh(model)
        for cfg in configs:
            cfg.model_id = model.id
        db.add_all(configs)
        db.flush()
    except IntegrityError as e:
        db.rollback()
        logger.exception("IntegrityError saving model: %s", str(e))
        for error in errors.err_msgs:
            if error in str(e):
                return _resp(400, False, errors.err_msgs[error])
        return _resp(400, False, "Model saving failed. Please recheck the model configs")

    try:
        model_path = os.path.join(MODEL_GENERATION_LOCATION, model_name + MODEL_GENERATION_TYPE)
        with open(model_path, "w+") as f:
            f.write(json.dumps(model_generated) + "\n")
        db.commit()
    except OSError:
        db.rollback()
        logger.exception("Error writing model file")
        return _resp(400, False, "Model validated but failed to save")

    logger.info("Model '%s' validated and saved successfully", model_name)
    return _resp(200, True, "Model validated and saved successfully", {"summary": _build_model_summary(keras_model)})


def update_training_config_service(
    db: Session, model_name: str, config: dict, project_id: uuid_pkg.UUID | None = None
) -> tuple:
    """Set training configuration on a previously saved model."""
    stmt = select(ModelBasic).where(ModelBasic.model_name == model_name)
    if project_id is not None:
        stmt = stmt.where(ModelBasic.project_id == project_id)
    model = db.exec(stmt).first()
    if not model:
        return _resp(404, False, "Model not found")

    problem_type_id = config["problem_type_id"]
    if problem_type_id in (ProblemType.CLASSIFICATION, ProblemType.IMAGE_CLASSIFICATION):
        loss = "sparse_categorical_crossentropy"
    else:
        loss = "mean_squared_error"

    model.file_id = config["file_id"]
    model.model_type = problem_type_id
    model.target_field = config.get("target_field")
    model.training_split = config["training_split"]
    model.optimizer = config["optimizer"]
    model.metric = config["metric"]
    model.epochs = config["epochs"]
    model.batch_size = config.get("batch_size", 32)
    model.loss = loss

    try:
        db.add(model)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        logger.exception("IntegrityError updating training config: %s", str(e))
        return _resp(400, False, "Failed to update training configuration")

    logger.info("Training config updated for model '%s'", model_name)
    return _resp(200, True, "Training configuration saved successfully")


def _verify_model_project(db: Session, model_name: str, project_id: uuid_pkg.UUID | None) -> tuple | None:
    """If project_id is provided, verify the model belongs to that project. Returns an error response or None."""
    if project_id is None:
        return None
    model = db.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()
    if not model or model.project_id != project_id:
        return _resp(404, False, "Model not found in this project")
    return None


def get_code_service(db: Session, model_name: str, project_id: uuid_pkg.UUID | None = None) -> tuple:
    """Generate a downloadable Python training script for the named model."""
    err = _verify_model_project(db, model_name, project_id)
    if err:
        return err
    python_code = generate_code(model_name, db)
    return {"content": python_code, "file_name": model_name + ".py"}, 200


def run_code_service(db: Session, model_name: str, project_id: uuid_pkg.UUID | None = None, loop=None) -> tuple:
    """Train a saved model and emit progress events via Socket.IO."""
    err = _verify_model_project(db, model_name, project_id)
    if err:
        return err
    model = db.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()
    if not model:
        return _resp(404, False, "Model not found")
    if model.file_id is None or model.epochs is None:
        return _resp(400, False, "Training configuration not set. Please configure training parameters first.")
    try:
        model_run(model_name, db, loop=loop)
        logger.info("Model '%s' training completed", model_name)
        return _resp(200, True, "Model executed successfully.")
    except Exception as e:
        logger.exception("Model run failed: %s", str(e))
        for error in errors.err_msgs:
            if error in str(e):
                return _resp(400, False, errors.err_msgs[error])
        return _resp(400, False, "Model running failed. Please recheck the model configs")


def get_model_graph_service(db: Session, model_name: str, project_id: uuid_pkg.UUID | None = None) -> tuple:
    """Retrieve the full ReactFlow graph for a saved model by unflattening its ModelConfigs."""
    stmt = select(ModelBasic).where(ModelBasic.model_name == model_name)
    if project_id is not None:
        stmt = stmt.where(ModelBasic.project_id == project_id)
    model = db.exec(stmt).first()
    if not model:
        return _resp(404, False, "Model not found")

    configs = db.exec(select(ModelConfigs).where(ModelConfigs.model_id == model.id)).all()
    if not configs:
        return _resp(404, False, "No graph configuration found for this model")

    graph = _unflatten_model_configs(configs)

    # Ensure every node has a position (auto-layout fallback)
    for i, node in enumerate(graph.get("nodes", [])):
        if "position" not in node:
            node["position"] = {"x": 100, "y": i * 200}

    return _resp(200, True, "Model graph retrieved successfully", {"model_name": model_name, "graph": graph})


def _unflatten_model_configs(configs: list[ModelConfigs]) -> dict:
    """Reconstruct a nested dict (with lists) from flattened ModelConfigs rows.

    Handles the dot-separated keys produced by flatten_json.flatten(), converting
    all-numeric-keyed levels back into lists. Extracts just the graph portion
    (nodes + edges) if configs were stored from a full validate request.
    """
    # Build the flat dict
    flat = {cfg.parameter: cfg.value for cfg in configs}

    # Rebuild nested structure
    root: dict = {}
    for key, value in flat.items():
        parts = key.split(".")
        current = root
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    # Convert all-numeric-keyed dicts to sorted lists (recursively)
    result = _numeric_dicts_to_lists(root)

    # Extract graph portion — configs from model_validate_service nest under "model"
    if "nodes" not in result and "model" in result and isinstance(result["model"], dict):
        result = result["model"]

    # Coerce position x/y to numbers for ReactFlow
    for node in result.get("nodes", []):
        if "position" in node and isinstance(node["position"], dict):
            for axis in ("x", "y"):
                val = node["position"].get(axis)
                if isinstance(val, str):
                    with contextlib.suppress(ValueError, AttributeError):
                        node["position"][axis] = int(val) if val.lstrip("-").isdigit() else float(val)

    return result


def _numeric_dicts_to_lists(obj):
    """Recursively convert dicts whose keys are all non-negative integers into lists."""
    if not isinstance(obj, dict):
        return obj

    converted = {k: _numeric_dicts_to_lists(v) for k, v in obj.items()}

    if converted and all(k.isdigit() for k in converted):
        max_idx = max(int(k) for k in converted)
        result = [None] * (max_idx + 1)
        for k, v in converted.items():
            result[int(k)] = v
        return result

    return converted


def get_available_model_list(
    db: Session, project_id: uuid_pkg.UUID | None = None, offset: int = 0, limit: int = 50
) -> tuple:
    """Return a paginated list of saved models with metadata."""
    base_filter = select(ModelBasic)
    if project_id is not None:
        base_filter = base_filter.where(ModelBasic.project_id == project_id)

    base_filter = base_filter.order_by(ModelBasic.created_on.desc())

    total = db.exec(select(func.count()).select_from(base_filter.subquery())).one()

    models = db.exec(base_filter.offset(offset).limit(limit)).all()

    data = [
        {
            "id": m.id,
            "model_name": m.model_name,
            "created_on": m.created_on.isoformat() if m.created_on else None,
            "epochs": m.epochs,
            "optimizer": m.optimizer,
            "metric": m.metric,
            "loss": m.loss,
            "training_split": m.training_split,
        }
        for m in models
    ]

    body = {"success": True, "message": "Model list generated successfully.", "data": data}
    body["pagination"] = {"total": total, "offset": offset, "limit": limit}
    return body, 200


def delete_model_service(db: Session, model_id: int) -> tuple:
    """Delete a model and its associated ModelConfigs (cascade), and remove the JSON file."""
    model = db.get(ModelBasic, model_id)
    if not model:
        return _resp(404, False, "Model not found")

    model_name = model.model_name
    try:
        db.delete(model)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to delete model id=%s", model_id)
        return _resp(500, False, "Failed to delete model")

    # Best-effort removal of the JSON file — do not fail if already gone
    model_path = os.path.join(MODEL_GENERATION_LOCATION, model_name + MODEL_GENERATION_TYPE)
    with contextlib.suppress(OSError):
        os.remove(model_path)

    logger.info("Model '%s' (id=%s) deleted successfully", model_name, model_id)
    return _resp(200, True, f"Model '{model_name}' deleted successfully")
