import json
import uuid as uuid_pkg
from typing import Any

import tensorflow as tf
from flatten_json import flatten
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
    return {"success": success, "message": message, "data": data}, status_code


def model_validate_service(db: Session, incoming: dict, project_id: uuid_pkg.UUID | None = None) -> tuple:
    model_generated = model_generation(model_params=incoming["model"])

    try:
        tf.keras.models.model_from_json(json.dumps(model_generated))
    except Exception as e:
        logger.error("Model validation error: %s", str(e))
        for error in errors.err_msgs:
            if error in str(e):
                return _resp(400, False, errors.err_msgs[error])
        return _resp(400, False, "Model validation failed. Please recheck the model and inputs")

    code = incoming[CODE]
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
        db.commit()
        db.refresh(model)
        for cfg in configs:
            cfg.model_id = model.id
        db.add_all(configs)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        logger.exception("IntegrityError saving model: %s", str(e))
        for error in errors.err_msgs:
            if error in str(e):
                return _resp(400, False, errors.err_msgs[error])
        return _resp(400, False, "Model saving failed. Please recheck the model configs")

    try:
        with open(
            MODEL_GENERATION_LOCATION + code[DL_MODEL][MODEL_NAME] + MODEL_GENERATION_TYPE,
            "w+",
        ) as f:
            f.write(json.dumps(model_generated) + "\n")
    except OSError:
        logger.exception("Error writing model file")
        # Cleanup orphaned database records
        db.delete(model)
        db.commit()
        return _resp(400, False, "Model validated but failed to save")

    return _resp(200, True, "Model Validation and saving successful")


def get_code_service(db: Session, model_name: str) -> tuple:
    python_code = generate_code(model_name, db)
    return {"content": python_code, "file_name": model_name + ".py"}, 200


def run_code_service(db: Session, model_name: str) -> tuple:
    try:
        model_run(model_name, db)
        return _resp(200, True, "Model executed successfully.")
    except Exception as e:
        logger.exception("Model run failed: %s", str(e))
        for error in errors.err_msgs:
            if error in str(e):
                return _resp(400, False, errors.err_msgs[error])
        return _resp(400, False, "Model running failed. Please recheck the model configs")


def get_available_model_list(db: Session, project_id: uuid_pkg.UUID | None = None) -> tuple:
    stmt = select(ModelBasic)
    if project_id is not None:
        stmt = stmt.where(ModelBasic.project_id == project_id)
    models = db.exec(stmt).all()
    data = [m.model_name for m in models]
    return _resp(200, True, "Model list generated successfully.", data)
