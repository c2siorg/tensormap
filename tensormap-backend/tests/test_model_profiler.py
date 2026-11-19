"""Unit and API tests for pre-training model architecture profiler and parameter inspector."""

import os

import tensorflow as tf
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models.ml import ModelBasic
from app.schemas.model_profiler import ModelComplexityProfile
from app.services.deep_learning import _validate_model_path
from app.services.model_profiler import profile_keras_model, profile_model_service


def test_profile_keras_model_dense():
    """Tests parameter and memory calculations for standard feed-forward networks."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(64, activation="relu", name="dense_1"),
            tf.keras.layers.Dense(10, activation="softmax", name="dense_out"),
        ]
    )
    profile = profile_keras_model(model, batch_size=32)

    assert isinstance(profile, ModelComplexityProfile)
    assert profile.total_layers == 2
    assert profile.total_params == 1354
    assert profile.trainable_params == 1354
    assert profile.non_trainable_params == 0
    assert profile.param_memory_mb > 0
    assert profile.estimated_activation_memory_mb > 0
    assert len(profile.layers) == 2
    assert profile.layers[0].layer_name == "dense_1"
    assert profile.layers[0].total_params == 704


def test_profile_keras_model_bottleneck_warning():
    """Tests detection of severe dimensional drops between adjacent intermediate layers."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(256,)),
            tf.keras.layers.Dense(256, name="wide_layer"),
            tf.keras.layers.Dense(4, name="bottleneck_layer"),
            tf.keras.layers.Dense(2, name="output_layer"),
        ]
    )
    profile = profile_keras_model(model)
    assert any(w.category == "extreme_bottleneck" for w in profile.warnings)
    bottleneck_warn = next(w for w in profile.warnings if w.category == "extreme_bottleneck")
    assert bottleneck_warn.layer_name == "bottleneck_layer"


def test_output_layer_compression_not_flagged_as_bottleneck():
    """Tests that ordinary compression into final classifier head is not flagged as bottleneck."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(128,)),
            tf.keras.layers.Dense(128, name="feature_layer"),
            tf.keras.layers.Dense(10, activation="softmax", name="output_classes"),
        ]
    )
    profile = profile_keras_model(model)
    # Output layer compression (128 -> 10) must not be flagged
    assert not any(w.category == "extreme_bottleneck" for w in profile.warnings)


def test_profile_keras_model_zero_trainable_warning():
    """Tests warning raised when model contains parameters but all are frozen."""
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(10,)), tf.keras.layers.Dense(32)])
    model.trainable = False
    profile = profile_keras_model(model)
    assert any(w.category == "zero_trainable" for w in profile.warnings)


def test_profile_keras_model_parameter_concentration():
    """Tests warning raised when a single layer holds >=85% of total model parameters."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(2, name="small_layer_1"),
            tf.keras.layers.Dense(1024, name="huge_concentrated_layer"),
            tf.keras.layers.Dropout(0.2, name="drop_layer"),
        ]
    )
    profile = profile_keras_model(model)
    assert any(w.category == "parameter_concentration" for w in profile.warnings)
    warn = next(w for w in profile.warnings if w.category == "parameter_concentration")
    assert warn.layer_name == "huge_concentrated_layer"


def test_profile_model_service_not_found(db_session: Session):
    """Tests 404 response when profiling a non-existent model ID."""
    body, status_code = profile_model_service(db_session, model_id=99999)
    assert status_code == 404
    assert body["success"] is False
    assert "Model not found" in body["message"]


def test_profile_model_service_missing_architecture_file_404(db_session: Session):
    """Tests 404 response when DB record exists but model architecture file is missing on disk."""
    model_name = "missing_arch_test_model"
    model_record = ModelBasic(model_name=model_name, model_type=1, epochs=5, batch_size=32)
    db_session.add(model_record)
    db_session.commit()
    db_session.refresh(model_record)

    body, status_code = profile_model_service(db_session, model_id=model_record.id)
    assert status_code == 404
    assert body["success"] is False
    assert "Model architecture file not found" in body["message"]


def test_profile_model_service_and_api(db_session: Session, client: TestClient):
    """Tests end-to-end service and GET endpoint execution using validated model file paths."""
    model_name = "test_profiler_model"
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(8,)), tf.keras.layers.Dense(16, name="d1")])
    model_path = _validate_model_path(model_name)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "w", encoding="utf-8") as f:
        f.write(model.to_json())

    try:
        model_record = ModelBasic(model_name=model_name, model_type=1, epochs=10, batch_size=32)
        db_session.add(model_record)
        db_session.commit()
        db_session.refresh(model_record)

        body, status = profile_model_service(db_session, model_id=model_record.id)
        assert status == 200
        assert body["success"] is True
        assert body["data"]["total_params"] == (8 * 16 + 16)

        # Valid batch_size
        resp = client.get(f"/api/v1/model/{model_record.id}/profile?batch_size=64")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

        # Batch size validation (ge=1, le=1024)
        resp_invalid_low = client.get(f"/api/v1/model/{model_record.id}/profile?batch_size=0")
        assert resp_invalid_low.status_code == 422

        resp_invalid_high = client.get(f"/api/v1/model/{model_record.id}/profile?batch_size=2000")
        assert resp_invalid_high.status_code == 422
    finally:
        if os.path.exists(model_path):
            os.remove(model_path)
