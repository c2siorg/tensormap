import contextlib
import copy
import json
import os
import re
import sys
import uuid as uuid_pkg
from datetime import UTC
from typing import Any

import numpy as np
import pandas as pd
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


# Maximum size (in bytes) for the serialised graph JSON stored in the DB.
_MAX_GRAPH_JSON_BYTES = 512 * 1024  # 512 KB


def _extract_graph(payload: dict) -> dict | None:
    """Extract the graph portion (nodes + edges) from a request payload.

    Both ``model_validate_service`` (which receives the full request body) and
    ``model_save_service`` (which receives only the ``model`` sub-object) call
    this helper so that ``graph_json`` always stores the same shape:
    ``{"nodes": [...], "edges": [...], "model_name": ...}``.
    """
    if payload is None:
        return None
    # If the payload wraps the graph under a "model" key, unwrap it.
    if "nodes" not in payload and "model" in payload and isinstance(payload["model"], dict):
        return payload["model"]
    return payload


def _validate_graph_size(graph: dict | None) -> str | None:
    """Return an error message if the serialised graph exceeds the size limit, else None."""
    if graph is None:
        return None
    raw = json.dumps(graph)
    size = sys.getsizeof(raw)
    if size > _MAX_GRAPH_JSON_BYTES:
        return f"Graph payload too large ({size} bytes > {_MAX_GRAPH_JSON_BYTES})"
    return None


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
    try:
        model_generated = model_generation(model_params=incoming["model"])
    except ValueError as e:
        return _resp(400, False, str(e))

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
        graph_json=_extract_graph(incoming),
    )

    size_err = _validate_graph_size(model.graph_json)
    if size_err:
        return _resp(413, False, size_err)

    configs = []
    params = flatten(incoming, separator=".")
    for param in params:
        if params[param] is not None:
            configs.append(ModelConfigs(parameter=param, value=str(params[param])))

    # Issue #6: write file BEFORE DB commit so DB and filesystem stay in sync.
    # If file write fails, DB is never touched. If DB fails, file is cleaned up.
    model_path = _validate_model_path(code[DL_MODEL][MODEL_NAME])  # Issue #1
    try:
        with open(model_path, "w+") as f:
            f.write(json.dumps(model_generated) + "\n")
        logger.info("Model JSON written to %s", model_path)
    except OSError:
        logger.exception("Error writing model file")
        return _resp(400, False, "Model validated but failed to save")

    try:
        db.add(model)
        db.flush()
        db.refresh(model)
        for cfg in configs:
            cfg.model_id = model.id
        db.add_all(configs)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        with contextlib.suppress(OSError):
            os.remove(model_path)
        logger.exception("IntegrityError saving model: %s", str(e))
        for error in errors.err_msgs:
            if error in str(e):
                return _resp(400, False, errors.err_msgs[error])
        return _resp(400, False, "Model saving failed. Please recheck the model configs")
    except Exception as e:
        db.rollback()
        with contextlib.suppress(OSError):
            os.remove(model_path)
        logger.exception("Unexpected error saving model: %s", str(e))
        return _resp(500, False, "Model saving failed. Please try again.")

    logger.info("Model '%s' validated and saved successfully", code[DL_MODEL][MODEL_NAME])
    return _resp(200, True, "Model Validation and saving successful", {"summary": _build_model_summary(keras_model)})


def model_save_service(db: Session, incoming: dict, model_name: str, project_id: uuid_pkg.UUID | None = None) -> tuple:
    """Validate a model graph with Keras and save architecture only (no training config)."""
    try:
        model_generated = model_generation(model_params=incoming)
    except ValueError as e:
        return _resp(400, False, str(e))

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

    graph_data = _extract_graph(incoming)
    size_err = _validate_graph_size(graph_data)
    if size_err:
        return _resp(413, False, size_err)

    model = ModelBasic(
        model_name=model_name,
        project_id=project_id,
        graph_json=graph_data,
    )

    configs = []
    params = flatten(incoming, separator=".")
    for param in params:
        if params[param] is not None:
            configs.append(ModelConfigs(parameter=param, value=str(params[param])))

    # Issue #6: write file BEFORE DB commit so DB and filesystem stay in sync.
    # If file write fails, DB is never touched. If DB fails, file is cleaned up.
    model_path = _validate_model_path(model_name)  # Issue #1
    try:
        with open(model_path, "w+") as f:
            f.write(json.dumps(model_generated) + "\n")
        logger.info("Model JSON written to %s", model_path)
    except OSError:
        logger.exception("Error writing model file")
        return _resp(400, False, "Model validated but failed to save")

    try:
        db.add(model)
        db.flush()
        db.refresh(model)
        for cfg in configs:
            cfg.model_id = model.id
        db.add_all(configs)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        with contextlib.suppress(OSError):
            os.remove(model_path)
        logger.exception("IntegrityError saving model: %s", str(e))
        for error in errors.err_msgs:
            if error in str(e):
                return _resp(400, False, errors.err_msgs[error])
        return _resp(400, False, "Model saving failed. Please recheck the model configs")
    except Exception as e:
        db.rollback()
        with contextlib.suppress(OSError):
            os.remove(model_path)
        logger.exception("Unexpected error saving model: %s", str(e))
        return _resp(500, False, "Model saving failed. Please try again.")

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
    try:
        python_code = generate_code(model_name, db)
    except ValueError as e:
        return _resp(400, False, str(e))
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
    """Retrieve the full ReactFlow graph for a saved model.

    If ``graph_json`` is populated (models created/saved after the column was
    added), the raw ReactFlow JSON is returned directly.  Older models fall back
    to reconstructing the graph from the flattened ``ModelConfigs`` rows.
    """
    stmt = select(ModelBasic).where(ModelBasic.model_name == model_name)
    if project_id is not None:
        stmt = stmt.where(ModelBasic.project_id == project_id)
    model = db.exec(stmt).first()
    if not model:
        return _resp(404, False, "Model not found")

    # Fast path: graph was stored directly in the JSON column
    if model.graph_json is not None:
        # Deep-copy to avoid mutating the ORM-tracked dict (which could trigger
        # an unintended UPDATE on the next flush/commit in the same session).
        graph = copy.deepcopy(model.graph_json)
        _apply_auto_layout(graph)
        return _resp(200, True, "Model graph retrieved successfully", {"model_name": model_name, "graph": graph})

    # Legacy path: reconstruct graph from flattened ModelConfigs
    configs = db.exec(select(ModelConfigs).where(ModelConfigs.model_id == model.id)).all()
    if not configs:
        return _resp(404, False, "No graph configuration found for this model")

    graph = _unflatten_model_configs(configs)
    _apply_auto_layout(graph)

    return _resp(200, True, "Model graph retrieved successfully", {"model_name": model_name, "graph": graph})


def _apply_auto_layout(graph: dict) -> None:
    """Assign default positions to nodes that lack one.

    This is a simple vertical-stack layout used as a fallback so that the
    ReactFlow canvas can render the graph without crashing.  For a more
    sophisticated layout (e.g. dagre), see:
    https://reactflow.dev/docs/examples/layout/dagre/

    TODO: Replace with a proper auto-layout algorithm (e.g. dagre) in a
    follow-up issue.
    """
    for i, node in enumerate(graph.get("nodes", [])):
        if "position" not in node:
            node["position"] = {"x": 100.0, "y": float(i * 200)}


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
    """Return a paginated list of saved models with basic info (backward compatible)."""
    base_filter = select(ModelBasic).order_by(ModelBasic.created_on.desc())
    if project_id is not None:
        base_filter = base_filter.where(ModelBasic.project_id == project_id)

    total = db.exec(select(func.count()).select_from(base_filter.subquery())).one()
    models = db.exec(base_filter.offset(offset).limit(limit)).all()

    data = [{"id": m.id, "model_name": m.model_name} for m in models]

    body = {"success": True, "message": "Model list generated successfully.", "data": data}
    body["pagination"] = {"total": total, "offset": offset, "limit": limit}
    return body, 200


def check_model_name_service(db: Session, model_name: str, project_id: uuid_pkg.UUID | None = None) -> tuple:
    """Check if a model name is available."""
    stmt = select(ModelBasic).where(ModelBasic.model_name == model_name)
    if project_id is not None:
        stmt = stmt.where(ModelBasic.project_id == project_id)
    existing = db.exec(stmt).first()

    if existing:
        return {"success": False, "message": "Model name already in use", "data": {"available": False}}, 200
    return {"success": True, "message": "Model name is available", "data": {"available": True}}, 200


def get_model_count_service(db: Session, project_id: uuid_pkg.UUID | None = None) -> tuple:
    """Get model count."""
    stmt = select(func.count(ModelBasic.id))
    if project_id is not None:
        stmt = stmt.where(ModelBasic.project_id == project_id)

    total = db.exec(stmt).one()
    return {"success": True, "message": "Model count retrieved", "data": {"count": total}}, 200


def get_training_history_service(
    db: Session, project_id: uuid_pkg.UUID | None = None, offset: int = 0, limit: int = 50
) -> tuple:
    """Return a paginated list of models with enriched metadata for training history view."""
    base_filter = select(ModelBasic).order_by(ModelBasic.created_on.desc())
    if project_id is not None:
        base_filter = base_filter.where(ModelBasic.project_id == project_id)

    total = db.exec(select(func.count()).select_from(base_filter.subquery())).one()
    models = db.exec(base_filter.offset(offset).limit(limit)).all()

    data = [
        {
            "id": m.id,
            "model_name": m.model_name,
            "created_on": m.created_on.replace(tzinfo=UTC).isoformat() if m.created_on else None,
            "epochs": m.epochs,
            "optimizer": m.optimizer,
            "metric": m.metric,
            "loss": m.loss,
            "training_split": m.training_split,
        }
        for m in models
    ]

    body = {"success": True, "message": "Training history generated successfully.", "data": data}
    body["pagination"] = {"total": total, "offset": offset, "limit": limit}
    return body, 200


def interpret_model_service(
    db: Session,
    model_name: str,
    file_id: uuid_pkg.UUID | None = None,
    project_id: uuid_pkg.UUID | None = None,
) -> tuple:
    """Generate interpretability metrics for a trained model.

    For classification: confusion matrix and per-class metrics.
    For regression: feature importance via permutation.
    """
    from sqlmodel import select

    from app.models.ml import DataFile

    stmt = select(ModelBasic).where(ModelBasic.model_name == model_name)
    if project_id is not None:
        stmt = stmt.where(ModelBasic.project_id == project_id)
    model_record = db.exec(stmt).first()

    if not model_record:
        return {"success": False, "message": f"Model '{model_name}' not found", "data": None}, 404

    model_path = _validate_model_path(model_name)
    if not os.path.exists(model_path):
        return {"success": False, "message": "Model file not found", "data": None}, 404

    try:
        loaded_model = tf.keras.models.load_model(model_path)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        return {"success": False, "message": f"Could not load model: {e}", "data": None}, 400

    problem_type = ProblemType(model_record.problem_type)

    if file_id:
        data_file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
        if not data_file:
            return {"success": False, "message": "Data file not found", "data": None}, 404

        try:
            df = pd.read_csv(data_file.file_name)
            X = df.drop(columns=[data_file.target_field], errors="ignore")
            y_true = df[data_file.target_field]

            if problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTI_CLASS_CLASSIFICATION):
                y_pred = loaded_model.predict(X.values.reshape(-1, X.shape[1]), verbose=0)
                if len(y_pred.shape) > 1:
                    y_pred_classes = y_pred.argmax(axis=-1)
                else:
                    y_pred_classes = (y_pred > 0.5).astype(int).flatten()

                cm = tf.math.confusion_matrix(y_true, y_pred_classes).numpy()
                result = {"confusion_matrix": cm.tolist(), "problem_type": str(problem_type)}

                labels = sorted(y_true.unique().tolist()) if hasattr(y_true, "unique") else list(range(cm.shape[0]))
                result["labels"] = labels

                report = {}
                for i, label in enumerate(labels):
                    tp = int(cm[i][i])
                    fp = int(cm[:, i].sum()) - tp
                    fn = int(cm[i, :].sum()) - tp
                    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
                    report[str(label)] = {
                        "precision": round(precision_val, 3),
                        "recall": round(recall_val, 3),
                        "f1": round(
                            2 * precision_val * recall_val / (precision_val + recall_val)
                            if (precision_val + recall_val) > 0
                            else 0,
                            3,
                        ),
                        "support": int((y_true == label).sum()),
                    }
                result["classification_report"] = report
            else:
                y_pred = loaded_model.predict(X.values.reshape(-1, X.shape[1]), verbose=0).flatten()
                baseline_error = np.mean((y_true - y_pred) ** 2)

                importance = []
                for col in X.columns:
                    X_permuted = X.copy()
                    X_permuted[col] = np.random.permutation(X_permuted[col])
                    y_permuted = loaded_model.predict(
                        X_permuted.values.reshape(-1, X_permuted.shape[1]), verbose=0
                    ).flatten()
                    perm_error = np.mean((y_true - y_permuted) ** 2)
                    importance.append({"feature": col, "importance": round(perm_error - baseline_error, 4)})

                importance.sort(key=lambda x: abs(x["importance"]), reverse=True)
                result = {"feature_importance": importance[:10], "problem_type": str(problem_type)}
        except Exception as e:
            return {"success": False, "message": f"Could not process data: {e}", "data": None}, 400
    else:
        return {"success": False, "message": "file_id required for interpretability", "data": None}, 400

    return {"success": True, "message": "Interpretability generated", "data": result}, 200


def delete_model_service(db: Session, model_id: int) -> tuple:
    """Delete a model and its associated ModelConfigs (cascade), and remove the JSON file."""
    model = db.get(ModelBasic, model_id)
    if not model:
        return _resp(404, False, "Model not found")

    model_name = model.model_name
    try:
        db.delete(model)
        db.commit()
    except IntegrityError:
        db.rollback()
        logger.exception("Failed to delete model id=%s", model_id)  # Issue #2: specific exception
        return _resp(500, False, "Failed to delete model")

    # Best-effort removal of the JSON file — do not fail if already gone
    model_path = _validate_model_path(model_name)  # Issue #1: path traversal guard
    with contextlib.suppress(OSError):
        os.remove(model_path)

    logger.info("Model '%s' (id=%s) deleted successfully", model_name, model_id)
    return _resp(200, True, f"Model '{model_name}' deleted successfully")


def _validate_model_path(model_name: str) -> str:
    path = os.path.realpath(os.path.join(MODEL_GENERATION_LOCATION, model_name + MODEL_GENERATION_TYPE))
    base_dir = os.path.realpath(MODEL_GENERATION_LOCATION)
    if not path.startswith(base_dir + os.sep) and path != base_dir:
        raise ValueError("Invalid model path")
    return path


def export_model_service(
    db: Session, model_name: str, export_format: str = "savedmodel", project_id: uuid_pkg.UUID | None = None
) -> tuple:
    stmt = select(ModelBasic).where(ModelBasic.model_name == model_name)
    if project_id is not None:
        stmt = stmt.where(ModelBasic.project_id == project_id)
    model = db.exec(stmt).first()

    if not model:
        return {"success": False, "message": f"Model '{model_name}' not found", "data": None}, 404

    model_path = _validate_model_path(model_name)
    if not os.path.exists(model_path):
        return {"success": False, "message": "Model file not found", "data": None}, 404

    try:
        loaded_model = tf.keras.models.load_model(model_path)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        return {"success": False, "message": f"Could not load model: {e}", "data": None}, 400

    export_dir = os.path.join(MODEL_GENERATION_LOCATION, f"{model_name}_export", export_format)
    os.makedirs(export_dir, exist_ok=True)

    try:
        if export_format == "savedmodel":
            saved_path = os.path.join(export_dir, model_name)
            loaded_model.save(saved_path)
            return {"success": True, "message": f"Exported to {saved_path}", "data": {"path": saved_path}}, 200

        if export_format == "tflite":
            converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            tflite_path = os.path.join(export_dir, f"{model_name}.tflite")
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            return {"success": True, "message": f"Exported to {tflite_path}", "data": {"path": tflite_path}}, 200

        if export_format == "onnx":
            try:
                import tf2onnx
            except ImportError:
                return {"success": False, "message": "ONNX not available - install tf2onnx", "data": None}, 501
            onnx_path = os.path.join(export_dir, f"{model_name}.onnx")
            try:
                tf2onnx.convert.from_keras(loaded_model, output_path=onnx_path)
            except Exception as e:
                return {"success": False, "message": f"ONNX failed: {e}", "data": None}, 400
            return {"success": True, "message": f"Exported to {onnx_path}", "data": {"path": onnx_path}}, 200

        return {"success": False, "message": f"Unsupported: {export_format}", "data": None}, 400
    except Exception as e:
        logger.error("Export failed: %s", e)
        return {"success": False, "message": f"Export failed: {e}", "data": None}, 500


def _validate_model_path(model_name: str) -> str:
    """Resolve model JSON path and guard against directory traversal (Issue #1).

    model_run.py already uses _helper_generate_json_model_file_location() for
    reads; this mirrors that protection for the write side in deep_learning.py.
    """
    path = os.path.realpath(os.path.join(MODEL_GENERATION_LOCATION, model_name + MODEL_GENERATION_TYPE))
    base_dir = os.path.realpath(MODEL_GENERATION_LOCATION)
    if not path.startswith(base_dir + os.sep) and path != base_dir:
        raise ValueError("Invalid model path: escapes model directory")
    return path
