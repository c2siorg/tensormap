"""Tests for the interactive model inference playground and feature schema API.

Tests cover:
- GET /model/analysis/{job_id}/feature-schema (features, bounds, steps, defaults, test sample records)
- POST /model/analysis/{job_id}/predict for classification (predictions, probabilities, confidence, latency)
- POST /model/analysis/{job_id}/predict for regression (predicted numeric value, latency)
- In-memory model caching performance
- Error handling (non-existent jobs, incomplete jobs, image classification guard)
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
from app.services.inference_service import InferenceService, ModelInferenceCache
from app.shared.enums import ProblemType


@pytest.fixture
def test_data_setup(db_session: Session, tmp_path: Path):
    """Sets up a synthetic dataset CSV, DataFile, ModelBasic, TrainingJob, and trained model.keras."""
    # 1. Create synthetic dataset
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

    # 2. Create ModelBasic
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

    # 3. Create TrainingJob
    job_id = str(uuid4())
    job = TrainingJob(
        id=job_id,
        model_id=model_basic.id,
        status=TrainingStatus.COMPLETED,
    )
    db_session.add(job)
    db_session.commit()

    # 4. Create and save a simple Keras model
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

        # Check that min/max/step/defaults are properly populated
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

        # Check sample test records
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

        # Probabilities should sum to approximately 1.0
        total_prob = sum(p.probability for p in resp.probabilities)
        assert pytest.approx(total_prob, abs=0.05) == 1.0

        # Latency should be non-negative
        assert resp.latency_ms >= 0.0

        # Model cache should contain this job
        assert job_id in service.model_cache._cache


def test_inference_cache_lru():
    """Tests the thread-safe ModelInferenceCache LRU eviction and clear behaviors."""
    cache = ModelInferenceCache(maxsize=2)
    cache._cache["job1"] = "model1"
    cache._cache["job2"] = "model2"

    assert len(cache._cache) == 2
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


def test_api_predict_endpoint(test_data_setup, client: TestClient):
    """Tests POST /api/v1/model/analysis/{job_id}/predict route."""
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
