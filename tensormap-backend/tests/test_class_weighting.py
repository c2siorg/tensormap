"""Unit and integration tests for cost-sensitive balanced class weighting."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from app.models.data import DataFile
from app.models.ml import ModelBasic
from app.services.code_generation import generate_code
from app.services.deep_learning import update_training_config_service
from app.services.model_run import (
    ModelRunError,
    _run,
    compute_balanced_class_weights,
    compute_image_balanced_class_weights,
)
from app.shared.enums import LossFunction, Metric, Optimizer, ProblemType


@pytest.fixture
def test_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


class TestComputeBalancedClassWeights:
    """Test suite for compute_balanced_class_weights."""

    def test_binary_imbalanced_weights(self):
        # 3 class 0, 1 class 1: total 4
        # w0 = 4 / (2 * 3) = 2/3 = 0.666667
        # w1 = 4 / (2 * 1) = 2.0
        y = [0, 0, 0, 1]
        weights = compute_balanced_class_weights(y)
        assert len(weights) == 2
        assert weights[0] == pytest.approx(0.666667, rel=1e-4)
        assert weights[1] == pytest.approx(2.0, rel=1e-4)

    def test_multiclass_weights_with_pandas_series(self):
        # Classes 0: 2, 1: 1, 2: 3. Total: 6 samples, 3 classes
        # w0 = 6 / (3 * 2) = 1.0
        # w1 = 6 / (3 * 1) = 2.0
        # w2 = 6 / (3 * 3) = 0.666667
        y = pd.Series([0, 0, 1, 2, 2, 2])
        weights = compute_balanced_class_weights(y)
        assert weights[0] == 1.0
        assert weights[1] == 2.0
        assert weights[2] == pytest.approx(0.666667, rel=1e-4)

    def test_one_hot_encoded_2d_array(self):
        # One-hot encoded matrix: 2 samples of class 0, 1 sample of class 1
        y_one_hot = np.array([[1, 0], [1, 0], [0, 1]])
        weights = compute_balanced_class_weights(y_one_hot)
        assert weights[0] == pytest.approx(3 / (2 * 2), rel=1e-4)
        assert weights[1] == pytest.approx(3 / (2 * 1), rel=1e-4)

    def test_single_class_raises_value_error(self):
        with pytest.raises(ValueError, match="at least 2 distinct classes"):
            compute_balanced_class_weights([0, 0, 0])

    def test_empty_array_raises_value_error(self):
        with pytest.raises(ValueError, match="empty target data"):
            compute_balanced_class_weights([])

    def test_none_input_raises_value_error(self):
        with pytest.raises(ValueError, match="target data is None"):
            compute_balanced_class_weights(None)


class TestComputeImageBalancedClassWeights:
    """Test suite for compute_image_balanced_class_weights."""

    def test_image_weights_calculated_correctly(self, tmp_path):
        # Create mock image dataset structure: class_a with 3 images, class_b with 1 image
        dataset_dir = tmp_path / "images"
        dir_a = dataset_dir / "class_a"
        dir_b = dataset_dir / "class_b"
        dir_a.mkdir(parents=True)
        dir_b.mkdir(parents=True)

        for i in range(3):
            (dir_a / f"img_{i}.jpg").write_text("fake image data")
        (dir_b / "img_0.png").write_text("fake image data")
        # Add a non-image file that should be ignored
        (dir_b / "notes.txt").write_text("metadata")

        weights = compute_image_balanced_class_weights(str(dataset_dir))
        # Alphabetical order: class_a is index 0, class_b is index 1
        # Total = 4, n_classes = 2
        # w0 = 4 / (2 * 3) = 0.666667, w1 = 4 / (2 * 1) = 2.0
        assert weights[0] == pytest.approx(0.666667, rel=1e-4)
        assert weights[1] == pytest.approx(2.0, rel=1e-4)

    def test_fewer_than_two_directories_raises(self, tmp_path):
        dataset_dir = tmp_path / "images_single"
        (dataset_dir / "only_one").mkdir(parents=True)
        (dataset_dir / "only_one" / "img.jpg").write_text("data")
        with pytest.raises(ValueError, match="at least 2 class directories"):
            compute_image_balanced_class_weights(str(dataset_dir))

    def test_nonexistent_directory_raises(self):
        with pytest.raises(ValueError, match="directory not found"):
            compute_image_balanced_class_weights("/nonexistent/path/123")


class TestUpdateTrainingConfigClassWeighting:
    """Test suite for update_training_config_service with class_weight_mode."""

    def test_saves_balanced_mode_on_classification(self, test_db):
        file_id = uuid4()
        data_file = DataFile(
            id=file_id,
            file_name="dataset.csv",
            file_type="csv",
            disk_name="dataset.csv",
            columns=["f1", "f2", "label"],
        )
        test_db.add(data_file)
        model = ModelBasic(
            model_name="test_clf_model",
            file_id=file_id,
            model_type=int(ProblemType.CLASSIFICATION),
        )
        test_db.add(model)
        test_db.commit()

        config = {
            "model_name": "test_clf_model",
            "file_id": str(file_id),
            "target_field": "label",
            "training_split": 80.0,
            "problem_type_id": int(ProblemType.CLASSIFICATION),
            "optimizer": Optimizer.ADAM.value,
            "metric": Metric.ACCURACY.value,
            "epochs": 10,
            "batch_size": 32,
            "loss": LossFunction.CATEGORICAL_CROSSENTROPY.value,
            "class_weight_mode": "balanced",
        }
        resp, status = update_training_config_service(test_db, "test_clf_model", config)
        assert status == 200
        assert resp["success"] is True

        updated = test_db.exec(ModelBasic.__table__.select().where(ModelBasic.model_name == "test_clf_model")).first()
        assert updated.class_weight_mode == "balanced"

    def test_rejects_invalid_class_weight_mode(self, test_db):
        file_id = uuid4()
        data_file = DataFile(
            id=file_id,
            file_name="dataset.csv",
            file_type="csv",
            disk_name="dataset.csv",
            columns=["f1", "label"],
        )
        test_db.add(data_file)
        model = ModelBasic(
            model_name="test_clf_model_invalid",
            file_id=file_id,
            model_type=int(ProblemType.CLASSIFICATION),
        )
        test_db.add(model)
        test_db.commit()

        config = {
            "model_name": "test_clf_model_invalid",
            "file_id": str(file_id),
            "target_field": "label",
            "training_split": 80.0,
            "problem_type_id": int(ProblemType.CLASSIFICATION),
            "optimizer": Optimizer.ADAM.value,
            "metric": Metric.ACCURACY.value,
            "epochs": 10,
            "loss": LossFunction.CATEGORICAL_CROSSENTROPY.value,
            "class_weight_mode": "unsupported_mode",
        }
        resp, status = update_training_config_service(test_db, "test_clf_model_invalid", config)
        assert status == 400
        assert "class_weight_mode must be 'balanced' or null" in resp["message"]

    def test_rejects_balanced_mode_on_regression(self, test_db):
        file_id = uuid4()
        data_file = DataFile(
            id=file_id,
            file_name="dataset.csv",
            file_type="csv",
            disk_name="dataset.csv",
            columns=["f1", "target"],
        )
        test_db.add(data_file)
        model = ModelBasic(
            model_name="test_reg_model",
            file_id=file_id,
            model_type=int(ProblemType.REGRESSION),
        )
        test_db.add(model)
        test_db.commit()

        config = {
            "model_name": "test_reg_model",
            "file_id": str(file_id),
            "target_field": "target",
            "training_split": 80.0,
            "problem_type_id": int(ProblemType.REGRESSION),
            "optimizer": Optimizer.ADAM.value,
            "metric": Metric.MSE.value,
            "epochs": 10,
            "loss": LossFunction.MEAN_SQUARED_ERROR.value,
            "class_weight_mode": "balanced",
        }
        resp, status = update_training_config_service(test_db, "test_reg_model", config)
        assert status == 400
        assert "only supported for classification" in resp["message"]


class TestCodeGenerationClassWeighting:
    """Test suite for code generation templates with class_weight_mode."""

    def test_csv_template_renders_class_weights_when_balanced(self, test_db, tmp_path):
        file_id = uuid4()
        data_file = DataFile(
            id=file_id,
            file_name="data.csv",
            file_type="csv",
            disk_name="data.csv",
            columns=["x1", "y"],
        )
        test_db.add(data_file)
        model = ModelBasic(
            model_name="codegen_balanced_model",
            file_id=file_id,
            model_type=int(ProblemType.CLASSIFICATION),
            target_field="y",
            training_split=80.0,
            optimizer="adam",
            metric="accuracy",
            epochs=5,
            batch_size=32,
            loss="categorical_crossentropy",
            class_weight_mode="balanced",
        )
        test_db.add(model)
        test_db.commit()

        code = generate_code("codegen_balanced_model", test_db)
        assert "class_weight=class_weights" in code
        assert "Compute balanced class weights" in code

    def test_csv_template_omits_class_weights_when_unweighted(self, test_db):
        file_id = uuid4()
        data_file = DataFile(
            id=file_id,
            file_name="data.csv",
            file_type="csv",
            disk_name="data.csv",
            columns=["x1", "y"],
        )
        test_db.add(data_file)
        model = ModelBasic(
            model_name="codegen_unweighted_model",
            file_id=file_id,
            model_type=int(ProblemType.CLASSIFICATION),
            target_field="y",
            training_split=80.0,
            optimizer="adam",
            metric="accuracy",
            epochs=5,
            batch_size=32,
            loss="categorical_crossentropy",
            class_weight_mode=None,
        )
        test_db.add(model)
        test_db.commit()

        code = generate_code("codegen_unweighted_model", test_db)
        assert "class_weight" not in code
        assert "Compute balanced class weights" not in code


class TestModelRunClassWeighting:
    """Test suite for _run passing class_weight to model.fit."""

    @patch("app.services.model_run.tf.keras.models.model_from_json")
    @patch("app.services.model_run._helper_generate_json_model_file_location")
    @patch("app.services.model_run._helper_generate_file_location")
    @patch("app.services.model_run.pd.read_csv")
    def test_run_passes_class_weight_to_fit_for_balanced_tabular(
        self, mock_read_csv, mock_gen_file, mock_gen_json, mock_model_from_json, test_db, tmp_path
    ):
        file_id = uuid4()
        data_file = DataFile(
            id=file_id,
            file_name="imbalanced.csv",
            file_type="csv",
            disk_name="imbalanced.csv",
            columns=["feat1", "target"],
        )
        test_db.add(data_file)
        model_obj = ModelBasic(
            model_name="balanced_run_model",
            file_id=file_id,
            model_type=int(ProblemType.CLASSIFICATION),
            target_field="target",
            training_split=80.0,
            optimizer="adam",
            metric="accuracy",
            epochs=2,
            batch_size=16,
            loss="sparse_categorical_crossentropy",
            class_weight_mode="balanced",
        )
        test_db.add(model_obj)
        test_db.commit()

        # Mock CSV data: 8 samples of class 0, 2 samples of class 1
        df = pd.DataFrame(
            {
                "feat1": [1.0] * 10,
                "target": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            }
        )
        mock_read_csv.return_value = df
        mock_gen_file.return_value = str(tmp_path / "imbalanced.csv")

        # Mock model JSON file
        json_file = tmp_path / "balanced_run_model.json"
        json_file.write_text('{"class_name": "Sequential", "config": {}}')
        mock_gen_json.return_value = str(json_file)

        mock_keras_model = MagicMock()
        mock_model_from_json.return_value = mock_keras_model

        _run("balanced_run_model", test_db)

        # Verify model.fit was called with class_weight dict
        assert mock_keras_model.fit.called
        call_kwargs = mock_keras_model.fit.call_args[1]
        assert "class_weight" in call_kwargs
        cw = call_kwargs["class_weight"]
        assert isinstance(cw, dict)
        assert 0 in cw and 1 in cw
        # Class 1 should have significantly higher weight than class 0
        assert cw[1] > cw[0]

    def test_run_raises_when_balanced_on_regression(self, test_db):
        model_obj = ModelBasic(
            model_name="regression_balanced_run",
            file_id=uuid4(),
            model_type=int(ProblemType.REGRESSION),
            target_field="val",
            training_split=80.0,
            optimizer="adam",
            metric="mse",
            epochs=2,
            batch_size=16,
            loss="mean_squared_error",
            class_weight_mode="balanced",
        )
        test_db.add(model_obj)
        test_db.commit()

        with (
            patch("app.services.model_run._prepare_training_data") as mock_prep,
            patch("app.services.model_run.pd.read_csv"),
            patch("app.services.model_run._helper_generate_file_location"),
        ):
            mock_prep.return_value = (np.array([1]), np.array([1]), np.array([]), np.array([]))
            with pytest.raises(ModelRunError, match="only supported for classification"):
                _run("regression_balanced_run", test_db)
