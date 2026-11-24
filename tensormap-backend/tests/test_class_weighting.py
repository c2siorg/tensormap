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


def test_compute_balanced_class_weights_one_hot_targets():
    """2D one-hot encoded targets yield per-class inverse-frequency weights."""
    # 3 classes: class 0 has 80 samples, class 1 has 15 samples, class 2 has 5 samples
    y_one_hot = np.zeros((100, 3))
    y_one_hot[:80, 0] = 1.0
    y_one_hot[80:95, 1] = 1.0
    y_one_hot[95:, 2] = 1.0

    weights = compute_balanced_class_weights(y_one_hot)
    assert weights is not None
    assert len(weights) == 3
    # N=100, C=3
    # w0 = 100 / (3 * 80) = 0.4167
    # w1 = 100 / (3 * 15) = 2.2222
    # w2 = 100 / (3 * 5) = 6.6667
    assert pytest.approx(weights[0], abs=1e-3) == 0.4167
    assert pytest.approx(weights[1], abs=1e-3) == 2.2222
    assert pytest.approx(weights[2], abs=1e-3) == 6.6667


def test_compute_balanced_class_weights_string_labels():
    """String categorical labels are indexed and produce balanced class weights."""
    y_strings = np.array(["cat"] * 70 + ["dog"] * 30)
    weights = compute_balanced_class_weights(y_strings)
    assert weights is not None
    assert len(weights) == 2
    # cat=0 (70 samples), dog=1 (30 samples)
    # w0 = 100 / (2 * 70) = 0.7143
    # w1 = 100 / (2 * 30) = 1.6667
    assert pytest.approx(weights[0], abs=1e-3) == 0.7143
    assert pytest.approx(weights[1], abs=1e-3) == 1.6667


def test_model_basic_has_class_weight_mode_column(db_session: Session):
    """Verify class_weight_mode column exists on model_basic table and is nullable."""
    from sqlalchemy import inspect

    inspector = inspect(db_session.bind)
    columns = {col["name"]: col for col in inspector.get_columns("model_basic")}
    assert "class_weight_mode" in columns
    assert columns["class_weight_mode"]["nullable"] is True


def test_update_training_config_rejects_invalid_class_weight_mode(db_session: Session):
    """Passing an unallowed string to class_weight_mode is rejected with 422 by schema validation."""
    import uuid as uuid_pkg

    from pydantic import ValidationError

    from app.schemas.deep_learning import TrainingConfigRequest

    with pytest.raises(ValidationError):
        TrainingConfigRequest(
            model_name="test_model",
            file_id=uuid_pkg.uuid4(),
            training_split=80.0,
            problem_type_id=1,
            optimizer=Optimizer.ADAM,
            metric=Metric.ACCURACY,
            epochs=10,
            class_weight_mode="arbitrary_invalid_mode",
        )


def test_run_tabular_model_passes_balanced_class_weights_to_fit():
    """_run computes balanced class weights and passes them to model.fit."""
    from unittest.mock import MagicMock, patch

    import pandas as pd

    from app.services.model_run import _run

    cfg = MagicMock()
    cfg.model_type = ProblemType.CLASSIFICATION
    cfg.model_name = "tabular_weighted_model"
    cfg.epochs = 2
    cfg.batch_size = 32
    cfg.optimizer = "adam"
    cfg.learning_rate = None
    cfg.loss = "categorical_crossentropy"
    cfg.metric = "accuracy"
    cfg.training_split = 80.0
    cfg.target_field = "label"
    cfg.file_id = "test-file-id"
    cfg.class_weight_mode = "balanced"

    mock_x_train = pd.DataFrame({"f1": [1.0] * 90 + [2.0] * 10})
    mock_y_train = pd.Series([0] * 90 + [1] * 10)
    mock_x_test = pd.DataFrame({"f1": [1.0, 2.0]})
    mock_y_test = pd.Series([0, 1])

    db = MagicMock()
    db.exec.return_value.first.return_value = cfg
    db.get.return_value = None

    mock_model = MagicMock()
    mock_model.stop_training = False
    with (
        patch("app.services.model_run._build_training_callbacks", return_value=[]),
        patch("app.services.model_run._helper_generate_file_location", return_value="/fake/data.csv"),
        patch("app.services.model_run.pd.read_csv", return_value=MagicMock()),
        patch(
            "app.services.model_run._prepare_training_data",
            return_value=(mock_x_train, mock_y_train, mock_x_test, mock_y_test),
        ),
        patch("builtins.open", MagicMock()),
        patch("app.services.model_run.tf.keras.models.model_from_json", return_value=mock_model),
        patch("app.services.model_run._emits_logits", return_value=False),
    ):
        _run("tabular_weighted_model", db, job_id=None)

    mock_model.fit.assert_called_once()
    fit_kwargs = mock_model.fit.call_args.kwargs
    assert "class_weight" in fit_kwargs
    weights = fit_kwargs["class_weight"]
    assert weights is not None
    assert pytest.approx(weights[0], abs=1e-3) == 0.5556
    assert pytest.approx(weights[1], abs=1e-3) == 5.0


def test_run_image_model_computes_class_weights_without_decoding(tmp_path):
    """_run computes class weights from class directory file counts for image classification."""
    from unittest.mock import MagicMock, patch

    from app.services.model_run import _run

    # Create dummy class directories
    c0 = tmp_path / "cats"
    c0.mkdir()
    for i in range(80):
        (c0 / f"img_{i}.jpg").write_text("dummy")
    c1 = tmp_path / "dogs"
    c1.mkdir()
    for i in range(20):
        (c1 / f"img_{i}.jpg").write_text("dummy")

    cfg = MagicMock()
    cfg.model_type = ProblemType.IMAGE_CLASSIFICATION
    cfg.model_name = "image_weighted_model"
    cfg.epochs = 2
    cfg.batch_size = 16
    cfg.optimizer = "adam"
    cfg.learning_rate = None
    cfg.loss = "sparse_categorical_crossentropy"
    cfg.metric = "accuracy"
    cfg.training_split = 80.0
    cfg.file_id = "test-image-id"
    cfg.class_weight_mode = "balanced"

    db = MagicMock()
    db.exec.return_value.first.return_value = cfg
    db.get.return_value = None

    mock_model = MagicMock()
    mock_model.stop_training = False
    with (
        patch("app.services.model_run._build_training_callbacks", return_value=[]),
        patch("app.services.model_run._helper_generate_file_location", return_value=str(tmp_path)),
        patch("app.services.model_run.tf.keras.utils.image_dataset_from_directory", return_value=MagicMock()),
        patch("builtins.open", MagicMock()),
        patch("app.services.model_run.tf.keras.models.model_from_json", return_value=mock_model),
        patch("app.services.model_run._emits_logits", return_value=False),
    ):
        _run("image_weighted_model", db, job_id=None)

    mock_model.fit.assert_called_once()
    fit_kwargs = mock_model.fit.call_args.kwargs
    assert "class_weight" in fit_kwargs
    weights = fit_kwargs["class_weight"]
    assert weights is not None
    # 100 total samples, 2 classes. Class 0: 100 / (2 * 80) = 0.625, Class 1: 100 / (2 * 20) = 2.5
    assert pytest.approx(weights[0], abs=1e-3) == 0.625
    assert pytest.approx(weights[1], abs=1e-3) == 2.5
