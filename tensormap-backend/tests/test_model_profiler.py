"""Unit and API tests for pre-training model architecture profiler and parameter inspector."""

from pathlib import Path

import tensorflow as tf
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models.ml import ModelBasic
from app.schemas.model_profiler import ModelComplexityProfile
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
    """Tests detection of severe dimensional drops between adjacent layers."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(256,)),
            tf.keras.layers.Dense(256, name="wide_layer"),
            tf.keras.layers.Dense(4, name="bottleneck_layer"),
        ]
    )
    profile = profile_keras_model(model)
    assert any(w.category == "extreme_bottleneck" for w in profile.warnings)
    bottleneck_warn = next(w for w in profile.warnings if w.category == "extreme_bottleneck")
    assert bottleneck_warn.layer_name == "bottleneck_layer"


def test_profile_keras_model_zero_trainable_warning():
    """Tests warning raised when model contains parameters but all are frozen."""
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(10,)), tf.keras.layers.Dense(32)])
    model.trainable = False
    profile = profile_keras_model(model)
    assert any(w.category == "zero_trainable" for w in profile.warnings)


def test_profile_model_service_not_found(db_session: Session):
    """Tests 404 response when profiling a non-existent model ID."""
    body, status_code = profile_model_service(db_session, model_id=99999)
    assert status_code == 404
    assert body["success"] is False
    assert "Model not found" in body["message"]


def test_profile_model_service_and_api(db_session: Session, client: TestClient):
    """Tests end-to-end service and GET endpoint execution using persisted model metadata."""
    model_name = "test_profiler_model"
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(8,)), tf.keras.layers.Dense(16, name="d1")])
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    json_file = data_dir / f"{model_name}.json"
    json_file.write_text(model.to_json())

    try:
        model_record = ModelBasic(model_name=model_name, model_type=1, epochs=10, batch_size=32)
        db_session.add(model_record)
        db_session.commit()
        db_session.refresh(model_record)

        body, status = profile_model_service(db_session, model_id=model_record.id)
        assert status == 200
        assert body["success"] is True
        assert body["data"]["total_params"] == (8 * 16 + 16)

        resp = client.get(f"/api/v1/model/{model_record.id}/profile?batch_size=64")
        assert resp.status_code == 200
        assert resp.json()["success"] is True
    finally:
        if json_file.exists():
            json_file.unlink()
