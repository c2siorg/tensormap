"""Tests for the interactive model inference playground and feature schema API.

Tests cover:
- GET /model/analysis/{job_id}/feature-schema (features, bounds, steps, defaults, test sample records)
- POST /model/analysis/{job_id}/predict for classification (predictions, probabilities, confidence, latency)
- POST /model/analysis/{job_id}/predict for regression (predicted numeric value, latency)
- Thread-safe LRU ModelInferenceCache (get_or_load, hits, move_to_end, eviction at maxsize, clear)
- Image classification guard returning 400
- Input validation (unknown features, missing features, invalid numeric values)
- Error handling (non-existent jobs, incomplete jobs)
"""

from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models.data import DataFile
from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus
from app.services.dataset_utils import prepare_tabular_dataset
from app.services.inference_service import InferenceService, ModelInferenceCache
from app.shared.enums import ProblemType


@pytest.fixture
def test_data_setup(db_session: Session, tmp_path: Path):
    """Sets up a synthetic classification dataset CSV, DataFile, ModelBasic, TrainingJob, and trained model.keras."""
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame(
        {
            "sepal_length": np.random.uniform(4.0, 8.0, n_samples),
            "sepal_width": np.random.uniform(2.0, 4.5, n_samples),
            "petal_length": np.random.uniform(1.0, 7.0, n_samples),
            "petal_width": np.random.uniform(0.1, 2.5, n_samples),
            "species": np.random.choice(["setosa", "versicolor", "virginica"], n_samples),
        }
    )

    upload_folder = tmp_path / "uploads"
    upload_folder.mkdir(parents=True, exist_ok=True)
    csv_file_name = f"dataset_{uuid4().hex[:8]}.csv"
    csv_path = upload_folder / csv_file_name
    df.to_csv(csv_path, index=False)

    data_file = DataFile(
        file_name="iris.csv",
        disk_name=csv_file_name,
        file_type="csv",
        total_rows=n_samples,
        total_columns=5,
    )
    db_session.add(data_file)
    db_session.commit()
    db_session.refresh(data_file)

    model_basic = ModelBasic(
        model_name="iris_classifier_test",
        file_id=data_file.id,
        target_field="species",
        model_type=ProblemType.CLASSIFICATION,
        training_split=80.0,
    )
    db_session.add(model_basic)
    db_session.commit()
    db_session.refresh(model_basic)

    job_id = str(uuid4())
    job = TrainingJob(
        id=job_id,
        model_id=model_basic.id,
        status=TrainingStatus.COMPLETED,
    )
    db_session.add(job)
    db_session.commit()

    import tensorflow as tf

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True, exist_ok=True)

    inputs = tf.keras.Input(shape=(4,))
    x = tf.keras.layers.Dense(8, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    keras_model.save(export_dir / "model.keras")

    return {
        "job_id": job_id,
        "data_file": data_file,
        "model_basic": model_basic,
        "export_dir": export_dir,
        "upload_folder": upload_folder,
        "df": df,
    }


@pytest.fixture
def regression_test_data_setup(db_session: Session, tmp_path: Path):
    """Sets up a synthetic regression dataset CSV, DataFile, ModelBasic, TrainingJob, and trained model.keras."""
    np.random.seed(42)
    n_samples = 80
    df = pd.DataFrame(
        {
            "feature_a": np.random.uniform(1.0, 10.0, n_samples),
            "feature_b": np.random.uniform(10.0, 50.0, n_samples),
            "price": np.random.uniform(100.0, 500.0, n_samples),
        }
    )

    upload_folder = tmp_path / "uploads_reg"
    upload_folder.mkdir(parents=True, exist_ok=True)
    csv_file_name = f"dataset_reg_{uuid4().hex[:8]}.csv"
    csv_path = upload_folder / csv_file_name
    df.to_csv(csv_path, index=False)

    data_file = DataFile(
        file_name="housing.csv",
        disk_name=csv_file_name,
        file_type="csv",
        total_rows=n_samples,
        total_columns=3,
    )
    db_session.add(data_file)
    db_session.commit()
    db_session.refresh(data_file)

    model_basic = ModelBasic(
        model_name="housing_regressor_test",
        file_id=data_file.id,
        target_field="price",
        model_type=ProblemType.REGRESSION,
        training_split=80.0,
    )
    db_session.add(model_basic)
    db_session.commit()
    db_session.refresh(model_basic)

    job_id = str(uuid4())
    job = TrainingJob(
        id=job_id,
        model_id=model_basic.id,
        status=TrainingStatus.COMPLETED,
    )
    db_session.add(job)
    db_session.commit()

    import tensorflow as tf

    export_dir = tmp_path / "exports_reg" / job_id
    export_dir.mkdir(parents=True, exist_ok=True)

    inputs = tf.keras.Input(shape=(2,))
    x = tf.keras.layers.Dense(4, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.compile(optimizer="adam", loss="mean_squared_error")
    keras_model.save(export_dir / "model.keras")

    return {
        "job_id": job_id,
        "data_file": data_file,
        "model_basic": model_basic,
        "export_dir": export_dir,
        "upload_folder": upload_folder,
        "df": df,
    }


def test_feature_schema_extraction(test_data_setup, db_session: Session):
    """Verifies that feature schema returns correct metadata, bounds, and sample rows."""
    job_id = test_data_setup["job_id"]
    upload_folder = str(test_data_setup["upload_folder"])

    with (
        patch("app.services.inference_service.get_settings") as mock_settings,
        patch("app.services.inference_service.EXPORTS_BASE", test_data_setup["export_dir"].parent),
    ):
        mock_settings.return_value.upload_folder = upload_folder
        service = InferenceService()
        schema = service.get_feature_schema(job_id, db_session)

        assert schema.job_id == job_id
        assert schema.target_field == "species"
        assert schema.problem_type == "classification"
        assert len(schema.features) == 4

        feature_names = [f.name for f in schema.features]
        assert "sepal_length" in feature_names
        assert "petal_width" in feature_names

        for feat in schema.features:
            assert feat.min is not None
            assert feat.max is not None
            assert feat.min <= feat.max
            assert feat.step is not None
            assert feat.step > 0
            assert feat.min <= feat.default_value <= feat.max

        assert schema.class_names is not None
        assert len(schema.class_names) == 3
        assert "setosa" in schema.class_names

        assert len(schema.sample_records) > 0
        first_sample = schema.sample_records[0]
        assert "sepal_length" in first_sample.features
        assert first_sample.actual_class_name in schema.class_names


def test_run_inference_classification(test_data_setup, db_session: Session):
    """Verifies classification inference execution, probability distribution, and latency."""
    job_id = test_data_setup["job_id"]
    upload_folder = str(test_data_setup["upload_folder"])

    with (
        patch("app.services.inference_service.get_settings") as mock_settings,
        patch("app.services.inference_service.EXPORTS_BASE", test_data_setup["export_dir"].parent),
    ):
        mock_settings.return_value.upload_folder = upload_folder
        service = InferenceService()

        sample_inputs = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }

        resp = service.run_inference(job_id, sample_inputs, db_session)

        assert resp.success is True
        assert resp.problem_type == "classification"
        assert resp.predicted_class in [0, 1, 2]
        assert resp.predicted_label is not None
        assert 0.0 <= resp.confidence <= 1.0
        assert resp.probabilities is not None
        assert len(resp.probabilities) == 3

        total_prob = sum(p.probability for p in resp.probabilities)
        assert pytest.approx(total_prob, abs=0.05) == 1.0
        assert resp.latency_ms >= 0.0

        # Model cache should contain this job
        assert job_id in service.model_cache._cache
        entry = service.model_cache._cache[job_id]
        assert entry.feature_names == ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def test_inference_cache_lru(tmp_path: Path):
    """Tests ModelInferenceCache LRU eviction, hit re-ordering, and clear behavior."""
    cache = ModelInferenceCache(maxsize=2)

    model_paths = []
    for i in range(3):
        p = tmp_path / f"model_{i}.keras"
        p.touch()
        model_paths.append(p)

    with patch("tensorflow.keras.models.load_model") as mock_load:
        mock_load.side_effect = lambda path: f"mock_model_{path.name}"

        # 1. Load job1 and job2 -> Cache: [job1, job2]
        entry1 = cache.get_or_load("job1", model_paths[0])
        cache.get_or_load("job2", model_paths[1])
        assert len(cache._cache) == 2
        assert list(cache._cache.keys()) == ["job1", "job2"]

        # 2. Access job1 again -> Cache hit moves job1 to end -> Cache: [job2, job1]
        entry1_again = cache.get_or_load("job1", model_paths[0])
        assert entry1_again is entry1
        assert list(cache._cache.keys()) == ["job2", "job1"]

        # 3. Load job3 -> Exceeds maxsize 2, evicts least recently used (job2)
        cache.get_or_load("job3", model_paths[2])
        assert len(cache._cache) == 2
        assert "job2" not in cache._cache
        assert list(cache._cache.keys()) == ["job1", "job3"]

        # 4. Clear cache
        cache.clear()
        assert len(cache._cache) == 0


def test_api_feature_schema_endpoint(test_data_setup, client: TestClient):
    """Tests GET /api/v1/model/analysis/{job_id}/feature-schema route."""
    job_id = test_data_setup["job_id"]
    upload_folder = str(test_data_setup["upload_folder"])

    with (
        patch("app.services.inference_service.get_settings") as mock_settings,
        patch("app.services.inference_service.EXPORTS_BASE", test_data_setup["export_dir"].parent),
    ):
        mock_settings.return_value.upload_folder = upload_folder
        response = client.get(f"/api/v1/model/analysis/{job_id}/feature-schema")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "features" in data
        assert len(data["features"]) == 4
        assert "sample_records" in data


def test_api_predict_classification_endpoint(test_data_setup, client: TestClient):
    """Tests POST /api/v1/model/analysis/{job_id}/predict route for classification."""
    job_id = test_data_setup["job_id"]
    upload_folder = str(test_data_setup["upload_folder"])

    with (
        patch("app.services.inference_service.get_settings") as mock_settings,
        patch("app.services.inference_service.EXPORTS_BASE", test_data_setup["export_dir"].parent),
    ):
        mock_settings.return_value.upload_folder = upload_folder
        payload = {
            "inputs": {
                "sepal_length": 6.0,
                "sepal_width": 3.0,
                "petal_length": 4.5,
                "petal_width": 1.5,
            }
        }
        response = client.post(f"/api/v1/model/analysis/{job_id}/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["problem_type"] == "classification"
        assert "confidence" in data
        assert "probabilities" in data
        assert "latency_ms" in data


def test_api_predict_regression_endpoint(regression_test_data_setup, client: TestClient):
    """Tests feature schema and live predict endpoints for regression models."""
    job_id = regression_test_data_setup["job_id"]
    upload_folder = str(regression_test_data_setup["upload_folder"])

    with (
        patch("app.services.inference_service.get_settings") as mock_settings,
        patch("app.services.inference_service.EXPORTS_BASE", regression_test_data_setup["export_dir"].parent),
    ):
        mock_settings.return_value.upload_folder = upload_folder

        # 1. Feature schema
        schema_resp = client.get(f"/api/v1/model/analysis/{job_id}/feature-schema")
        assert schema_resp.status_code == 200
        schema_data = schema_resp.json()
        assert schema_data["problem_type"] == "regression"
        assert schema_data["target_field"] == "price"
        assert len(schema_data["features"]) == 2

        # 2. Predict
        payload = {
            "inputs": {
                "feature_a": 5.5,
                "feature_b": 25.0,
            }
        }
        pred_resp = client.post(f"/api/v1/model/analysis/{job_id}/predict", json=payload)
        assert pred_resp.status_code == 200
        pred_data = pred_resp.json()
        assert pred_data["success"] is True
        assert pred_data["problem_type"] == "regression"
        assert isinstance(pred_data["predicted_value"], float)
        assert pred_data["latency_ms"] >= 0.0


def test_image_classification_guard_returns_400(test_data_setup, db_session: Session, client: TestClient):
    """Tests that image classification jobs reject tabular inference with HTTP 400."""
    model_basic = test_data_setup["model_basic"]
    model_basic.model_type = ProblemType.IMAGE_CLASSIFICATION
    db_session.add(model_basic)
    db_session.commit()

    job_id = test_data_setup["job_id"]

    # 1. GET /feature-schema returns 400
    resp_schema = client.get(f"/api/v1/model/analysis/{job_id}/feature-schema")
    assert resp_schema.status_code == 400
    assert "not available for image classification" in resp_schema.json()["detail"]

    # 2. POST /predict returns 400
    payload = {"inputs": {"sepal_length": 5.0}}
    resp_pred = client.post(f"/api/v1/model/analysis/{job_id}/predict", json=payload)
    assert resp_pred.status_code == 400
    assert "not available for image classification" in resp_pred.json()["detail"]


def test_predict_input_validation_errors(test_data_setup, client: TestClient):
    """Tests strict validation: unknown keys, missing keys, and invalid numeric values."""
    job_id = test_data_setup["job_id"]
    upload_folder = str(test_data_setup["upload_folder"])

    with (
        patch("app.services.inference_service.get_settings") as mock_settings,
        patch("app.services.inference_service.EXPORTS_BASE", test_data_setup["export_dir"].parent),
    ):
        mock_settings.return_value.upload_folder = upload_folder

        # 1. Unknown feature provided
        bad_payload_unknown = {
            "inputs": {
                "sepal_length": 6.0,
                "sepal_width": 3.0,
                "petal_length": 4.5,
                "petal_width": 1.5,
                "bogus_feature": 99.0,
            }
        }
        resp = client.post(f"/api/v1/model/analysis/{job_id}/predict", json=bad_payload_unknown)
        assert resp.status_code == 400
        msg = resp.json().get("message") or resp.json().get("detail", "")
        assert "Unknown feature" in msg

        # 2. Missing feature provided
        bad_payload_missing = {
            "inputs": {
                "sepal_length": 6.0,
                "sepal_width": 3.0,
            }
        }
        resp = client.post(f"/api/v1/model/analysis/{job_id}/predict", json=bad_payload_missing)
        assert resp.status_code == 400
        msg = resp.json().get("message") or resp.json().get("detail", "")
        assert "Missing required feature" in msg

        # 3. Non-numeric value provided
        bad_payload_non_numeric = {
            "inputs": {
                "sepal_length": "not-a-number",
                "sepal_width": 3.0,
                "petal_length": 4.5,
                "petal_width": 1.5,
            }
        }
        resp = client.post(f"/api/v1/model/analysis/{job_id}/predict", json=bad_payload_non_numeric)
        assert resp.status_code == 422


def test_predict_non_existent_job(client: TestClient):
    """Tests 404 response when job does not exist."""
    fake_job_id = str(uuid4())
    response = client.get(f"/api/v1/model/analysis/{fake_job_id}/feature-schema")
    assert response.status_code == 404

    payload = {"inputs": {"f1": 1.0}}
    response = client.post(f"/api/v1/model/analysis/{fake_job_id}/predict", json=payload)
    assert response.status_code == 404


def test_predict_incomplete_job(test_data_setup, db_session: Session, client: TestClient):
    """Tests 400 response when job is not completed."""
    job_id = str(uuid4())
    job = TrainingJob(
        id=job_id,
        model_id=test_data_setup["model_basic"].id,
        status=TrainingStatus.RUNNING,
    )
    db_session.add(job)
    db_session.commit()

    response = client.get(f"/api/v1/model/analysis/{job_id}/feature-schema")
    assert response.status_code == 400

    payload = {"inputs": {}}
    response = client.post(f"/api/v1/model/analysis/{job_id}/predict", json=payload)
    assert response.status_code == 400


class TestPrepareTabularDatasetDirect:
    """Direct tests for prepare_tabular_dataset covering edge cases, steps, and targets."""

    def test_prepare_tabular_dataset_excludes_non_numeric_columns(self):
        df = pd.DataFrame(
            {
                "feat_num1": [1.0, 2.0, 3.0, 4.0],
                "feat_str": ["cat", "dog", "bird", "fish"],
                "target": [0, 1, 0, 1],
            }
        )
        ds = prepare_tabular_dataset(df, target_field="target")
        assert "feat_num1" in ds["feature_names"]
        assert "feat_str" not in ds["feature_names"]
        assert list(ds["X_df"].columns) == ["feat_num1", "feat_str"]

    def test_prepare_tabular_dataset_int_vs_float_step_logic(self):
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3, 4, 5],
                "float_col": [0.1, 0.5, 0.9, 1.2, 2.0],
                "target": [10, 20, 10, 20, 10],
            }
        )
        ds = prepare_tabular_dataset(df, target_field="target")
        assert ds["feature_dtypes"]["int_col"] == "int"
        assert ds["feature_steps"]["int_col"] == 1.0

        assert ds["feature_dtypes"]["float_col"] == "float"
        assert ds["feature_steps"]["float_col"] > 0
        assert ds["feature_steps"]["float_col"] != 1.0

    def test_prepare_tabular_dataset_nan_only_column_handled(self):
        # Note: dropna removes rows with nan, so test non-convertible empty strings
        df_no_nan_rows = pd.DataFrame(
            {
                "valid_feat": [1.0, 2.0, 3.0],
                "empty_str_col": ["", "", ""],
                "target": [0, 1, 0],
            }
        )
        ds = prepare_tabular_dataset(df_no_nan_rows, target_field="target")
        assert "valid_feat" in ds["feature_names"]
        assert "empty_str_col" not in ds["feature_names"]

    def test_prepare_tabular_dataset_non_zero_based_labels(self):
        df = pd.DataFrame(
            {
                "feat": [1.0, 2.0, 3.0, 4.0],
                "target": [1, 2, 3, 2],
            }
        )
        ds = prepare_tabular_dataset(df, target_field="target")
        assert ds["class_names"] == ["1", "2", "3"]

    def test_prepare_tabular_dataset_string_categorical_target(self):
        df = pd.DataFrame(
            {
                "feat": [10.0, 20.0, 30.0],
                "species": ["setosa", "versicolor", "setosa"],
            }
        )
        ds = prepare_tabular_dataset(df, target_field="species")
        assert ds["class_names"] == ["setosa", "versicolor"]
        assert list(ds["y_encoded"]) == [0, 1, 0] or list(ds["y_encoded"]) == [0, 1, 0]


class TestSampleRecordGroundTruthMapping:
    """Verifies that sample records correctly map non-zero and string labels to prediction class indices."""

    def test_sample_record_maps_non_zero_based_labels(self, db_session: Session, tmp_path: Path):
        # Create dataset with non-zero labels {1, 2}
        csv_path = tmp_path / "non_zero.csv"
        df = pd.DataFrame(
            {
                "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "target": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            }
        )
        df.to_csv(csv_path, index=False)

        file_id = uuid4()
        data_file = DataFile(
            id=file_id,
            file_name="non_zero.csv",
            file_type="csv",
            disk_name="non_zero.csv",
        )
        db_session.add(data_file)
        db_session.commit()
        db_session.refresh(data_file)

        model = ModelBasic(
            model_name=f"non_zero_model_{uuid4().hex[:6]}",
            file_id=data_file.id,
            target_field="target",
            training_split=50.0,
            epochs=1,
            batch_size=2,
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metric="accuracy",
        )
        db_session.add(model)
        db_session.commit()
        db_session.refresh(model)

        job = TrainingJob(
            id="job-non-zero",
            model_id=model.id,
            status=TrainingStatus.COMPLETED,
        )
        db_session.add(job)
        db_session.commit()

        svc = InferenceService()
        with patch("app.services.inference_service.get_settings") as mock_settings:
            mock_settings.return_value.upload_folder = str(tmp_path)
            schema = svc.get_feature_schema("job-non-zero", db_session)

            assert schema.class_names == ["1", "2"]
            assert len(schema.sample_records) > 0
            for record in schema.sample_records:
                assert record.actual_class_name in ("1", "2")
                # actual_class must be the index in class_names (0 or 1), NOT silently forced to 0
                expected_idx = schema.class_names.index(record.actual_class_name)
                assert record.actual_class == expected_idx

    def test_sample_record_maps_string_labels(self, db_session: Session, tmp_path: Path):
        csv_path = tmp_path / "strings.csv"
        df = pd.DataFrame(
            {
                "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "category": ["alpha", "beta", "alpha", "beta", "alpha", "beta", "alpha", "beta", "alpha", "beta"],
            }
        )
        df.to_csv(csv_path, index=False)

        file_id = uuid4()
        data_file = DataFile(
            id=file_id,
            file_name="strings.csv",
            file_type="csv",
            disk_name="strings.csv",
        )
        db_session.add(data_file)
        db_session.commit()
        db_session.refresh(data_file)

        model = ModelBasic(
            model_name=f"string_model_{uuid4().hex[:6]}",
            file_id=data_file.id,
            target_field="category",
            training_split=50.0,
            epochs=1,
            batch_size=2,
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metric="accuracy",
        )
        db_session.add(model)
        db_session.commit()
        db_session.refresh(model)

        job = TrainingJob(
            id="job-strings",
            model_id=model.id,
            status=TrainingStatus.COMPLETED,
        )
        db_session.add(job)
        db_session.commit()

        svc = InferenceService()
        with patch("app.services.inference_service.get_settings") as mock_settings:
            mock_settings.return_value.upload_folder = str(tmp_path)
            schema = svc.get_feature_schema("job-strings", db_session)

            assert schema.class_names == ["alpha", "beta"]
            assert len(schema.sample_records) > 0
            for record in schema.sample_records:
                assert record.actual_class_name in ("alpha", "beta")
                expected_idx = schema.class_names.index(record.actual_class_name)
                assert record.actual_class == expected_idx
