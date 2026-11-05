"""Tests for cost-sensitive class imbalance weighting and template code generation."""

import numpy as np
import pytest
from sqlmodel import Session

from app.models.data import DataFile
from app.models.ml import ModelBasic
from app.services.code_generation import generate_code
from app.services.deep_learning import update_training_config_service
from app.services.model_run import compute_balanced_class_weights
from app.shared.enums import LossFunction, Metric, Optimizer, ProblemType


def test_compute_balanced_class_weights_balanced_binary():
    """Equal class counts yield unit weights (1.0 for each class)."""
    y = np.array([0] * 50 + [1] * 50)
    weights = compute_balanced_class_weights(y)
    assert weights is not None
    assert weights[0] == 1.0
    assert weights[1] == 1.0


def test_compute_balanced_class_weights_imbalanced_binary():
    """Imbalanced 90:10 class distribution gives higher weight to minority class."""
    y = np.array([0] * 90 + [1] * 10)
    weights = compute_balanced_class_weights(y)
    assert weights is not None
    # Class 0: 100 / (2 * 90) = 0.5556
    # Class 1: 100 / (2 * 10) = 5.0
    assert pytest.approx(weights[0], abs=1e-3) == 0.5556
    assert pytest.approx(weights[1], abs=1e-3) == 5.0


def test_compute_balanced_class_weights_multiclass():
    """Multi-class imbalance (100, 20, 10 samples) assigns inversely proportional weights."""
    y = np.array([0] * 100 + [1] * 20 + [2] * 10)
    weights = compute_balanced_class_weights(y)
    assert weights is not None
    assert len(weights) == 3
    # Total samples = 130, classes = 3
    # w0 = 130 / (3 * 100) = 0.4333
    # w1 = 130 / (3 * 20) = 2.1667
    # w2 = 130 / (3 * 10) = 4.3333
    assert pytest.approx(weights[0], abs=1e-3) == 0.4333
    assert pytest.approx(weights[1], abs=1e-3) == 2.1667
    assert pytest.approx(weights[2], abs=1e-3) == 4.3333


def test_compute_balanced_class_weights_edge_cases():
    """Single class or empty input gracefully returns None."""
    assert compute_balanced_class_weights(np.array([1, 1, 1, 1])) is None
    assert compute_balanced_class_weights(np.array([])) is None


def test_update_training_config_saves_class_weight_mode(db_session: Session):
    """update_training_config_service persists class_weight_mode to ModelBasic."""
    data_file = DataFile(
        file_name="imbalanced.csv",
        disk_name="imbalanced.csv",
        file_type="csv",
        row_count=100,
        columns=["x1", "label"],
    )
    db_session.add(data_file)
    db_session.commit()
    db_session.refresh(data_file)

    model = ModelBasic(
        model_name="test_imbalance_model",
        file_id=data_file.id,
        target_field="label",
        training_split=80.0,
        optimizer="adam",
        metric="accuracy",
        epochs=10,
        model_type=ProblemType.CLASSIFICATION,
    )
    db_session.add(model)
    db_session.commit()

    config = {
        "model_name": "test_imbalance_model",
        "file_id": data_file.id,
        "target_field": "label",
        "training_split": 80.0,
        "problem_type_id": ProblemType.CLASSIFICATION,
        "optimizer": Optimizer.ADAM,
        "metric": Metric.ACCURACY,
        "epochs": 15,
        "batch_size": 32,
        "loss": LossFunction.CATEGORICAL_CROSSENTROPY,
        "class_weight_mode": "balanced",
    }

    body, status = update_training_config_service(db_session, "test_imbalance_model", config)
    assert status == 200
    assert body["success"] is True

    db_session.refresh(model)
    assert model.class_weight_mode == "balanced"
    assert model.epochs == 15


def test_code_generation_includes_class_weights_when_balanced(db_session: Session):
    """Generated script includes balanced class weight calculation for classification template."""
    data_file = DataFile(
        file_name="imbalanced.csv",
        disk_name="imbalanced.csv",
        file_type="csv",
        row_count=100,
        columns=["x1", "label"],
    )
    db_session.add(data_file)
    db_session.commit()
    db_session.refresh(data_file)

    model = ModelBasic(
        model_name="test_codegen_weighted_model",
        file_id=data_file.id,
        target_field="label",
        training_split=80.0,
        optimizer="adam",
        metric="accuracy",
        epochs=10,
        model_type=ProblemType.CLASSIFICATION,
        class_weight_mode="balanced",
    )
    db_session.add(model)
    db_session.commit()

    code = generate_code("test_codegen_weighted_model", db_session)
    assert 'class_weight_mode = "balanced"' in code
    assert "class_weight=class_weights" in code


def test_code_generation_omits_class_weights_when_unweighted(db_session: Session):
    """Generated script sets class_weight_mode to None when unweighted."""
    data_file = DataFile(
        file_name="balanced.csv",
        disk_name="balanced.csv",
        file_type="csv",
        row_count=100,
        columns=["x1", "label"],
    )
    db_session.add(data_file)
    db_session.commit()
    db_session.refresh(data_file)

    model = ModelBasic(
        model_name="test_codegen_unweighted_model",
        file_id=data_file.id,
        target_field="label",
        training_split=80.0,
        optimizer="adam",
        metric="accuracy",
        epochs=10,
        model_type=ProblemType.CLASSIFICATION,
        class_weight_mode=None,
    )
    db_session.add(model)
    db_session.commit()

    code = generate_code("test_codegen_unweighted_model", db_session)
    assert 'class_weight_mode = "None"' in code
    assert "class_weights = None" in code
