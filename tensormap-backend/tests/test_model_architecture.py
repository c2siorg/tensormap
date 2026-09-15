"""Tests for model architecture endpoint with parameter count (Week 8)."""

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.main import app
from app.models.ml import ModelBasic

client = TestClient(app)


def test_get_model_architecture_basic(db_session: Session):
    """GET /model/architecture/{model_id} should return model details."""
    # Create model
    graph_ir = {
        "nodes": [
            {"node_params": {"layer_type": "input", "shape": [28, 28, 1]}},
            {"node_params": {"layer_type": "dense", "units": 128, "activation": "relu"}},
            {"node_params": {"layer_type": "dense", "units": 10, "activation": "softmax"}},
        ]
    }
    model = ModelBasic(id=1, model_name="mnist_model", graph_ir=graph_ir)
    db_session.add(model)
    db_session.commit()

    # Request without stats
    response = client.get("/api/v1/model/architecture/1")
    assert response.status_code == 200

    body = response.json()
    assert body["success"] is True
    data = body["data"]
    assert data["model_id"] == 1
    assert data["model_name"] == "mnist_model"
    assert data["graph_ir"] == graph_ir
    assert "param_count" not in data
    assert "size_mb" not in data


def test_get_model_architecture_with_stats(db_session: Session):
    """GET /model/architecture/{model_id}?include_stats=true should include param count and size."""
    # Create a simple Dense model
    graph_ir = {
        "nodes": [
            {"node_params": {"layer_type": "input", "shape": [10]}},
            {"node_params": {"layer_type": "dense", "units": 64}},
            {"node_params": {"layer_type": "dense", "units": 32}},
            {"node_params": {"layer_type": "dense", "units": 3}},
        ]
    }
    model = ModelBasic(id=1, model_name="test_model", graph_ir=graph_ir)
    db_session.add(model)
    db_session.commit()

    # Request with stats
    response = client.get("/api/v1/model/architecture/1?include_stats=true")
    assert response.status_code == 200

    body = response.json()
    assert body["success"] is True
    data = body["data"]
    assert "param_count" in data
    assert "size_mb" in data

    # Verify param count is reasonable
    # Input(10) -> Dense(64): 10*64 + 64 = 704
    # Dense(64) -> Dense(32): 64*32 + 32 = 2080
    # Dense(32) -> Dense(3): 32*3 + 3 = 99
    # Total: 704 + 2080 + 99 = 2883
    assert data["param_count"] == 2883

    # Size should be roughly param_count * 4 bytes / 1024^2
    expected_size_mb = (2883 * 4) / (1024 * 1024)
    assert abs(data["size_mb"] - expected_size_mb) < 0.01


def test_get_model_architecture_lstm(db_session: Session):
    """Parameter estimation should work for LSTM models."""
    graph_ir = {
        "nodes": [
            {"node_params": {"layer_type": "input", "shape": [10, 128]}},
            {"node_params": {"layer_type": "lstm", "units": 64}},
            {"node_params": {"layer_type": "dense", "units": 3}},
        ]
    }
    model = ModelBasic(id=1, model_name="lstm_model", graph_ir=graph_ir)
    db_session.add(model)
    db_session.commit()

    response = client.get("/api/v1/model/architecture/1?include_stats=true")
    assert response.status_code == 200

    data = response.json()["data"]
    assert "param_count" in data
    # LSTM has 4 gates: 4 * (input_dim * units + units * units + units)
    # 4 * (128 * 64 + 64 * 64 + 64) = 4 * (8192 + 4096 + 64) = 49408
    # Plus Dense: 64 * 3 + 3 = 195
    # Total: 49408 + 195 = 49603
    assert data["param_count"] == 49603


def test_get_model_architecture_conv2d(db_session: Session):
    """Parameter estimation should work for Conv2D models."""
    graph_ir = {
        "nodes": [
            {"node_params": {"layer_type": "input", "shape": [28, 28, 1]}},
            {"node_params": {"layer_type": "conv2d", "filters": 32, "kernel_size": [3, 3]}},
            {"node_params": {"layer_type": "dense", "units": 10}},
        ]
    }
    model = ModelBasic(id=1, model_name="cnn_model", graph_ir=graph_ir)
    db_session.add(model)
    db_session.commit()

    response = client.get("/api/v1/model/architecture/1?include_stats=true")
    assert response.status_code == 200

    data = response.json()["data"]
    assert "param_count" in data
    # Conv2D: kernel_h * kernel_w * input_channels * filters + bias
    # Input shape is [28, 28, 1] but we track last dim as prev_shape
    # After input, prev_shape = 1
    # 3 * 3 * 1 * 32 + 32 = 288 + 32 = 320
    # Dense: 32 * 10 + 10 = 330
    # Total: 320 + 330 = 650
    assert data["param_count"] == 650


def test_get_model_architecture_not_found(db_session: Session):
    """Should return 404 for non-existent model."""
    response = client.get("/api/v1/model/architecture/99999")
    assert response.status_code == 404
    assert response.json()["success"] is False


def test_get_model_architecture_no_graph_ir(db_session: Session):
    """Should handle models without graph_ir gracefully."""
    model = ModelBasic(id=1, model_name="legacy_model", graph_ir=None)
    db_session.add(model)
    db_session.commit()

    # Without stats should work
    response = client.get("/api/v1/model/architecture/1")
    assert response.status_code == 200
    assert response.json()["data"]["graph_ir"] is None

    # With stats should not crash (just no stats)
    response = client.get("/api/v1/model/architecture/1?include_stats=true")
    assert response.status_code == 200
    # May or may not include param_count (depends on implementation)


def test_estimate_param_count_embedding(db_session: Session):
    """Parameter estimation should work for Embedding layers."""
    graph_ir = {
        "nodes": [
            {"node_params": {"layer_type": "embedding", "input_dim": 10000, "output_dim": 128}},
            {"node_params": {"layer_type": "lstm", "units": 64}},
            {"node_params": {"layer_type": "dense", "units": 1}},
        ]
    }
    model = ModelBasic(id=1, model_name="embedding_model", graph_ir=graph_ir)
    db_session.add(model)
    db_session.commit()

    response = client.get("/api/v1/model/architecture/1?include_stats=true")
    assert response.status_code == 200

    data = response.json()["data"]
    # Embedding: 10000 * 128 = 1,280,000
    # LSTM: 4 * (128 * 64 + 64 * 64 + 64) = 49,408
    # Dense: 64 * 1 + 1 = 65
    # Total: 1,329,473
    assert data["param_count"] == 1329473


def test_estimate_param_count_scalar_shape(db_session: Session):
    """The IR schema stores input shape as a scalar int (e.g. 10), not a list.
    The first Dense layer's weights must still be counted."""
    graph_ir = {
        "nodes": [
            {"node_params": {"layer_type": "input", "shape": 10}},
            {"node_params": {"layer_type": "dense", "units": 64}},
            {"node_params": {"layer_type": "dense", "units": 32}},
            {"node_params": {"layer_type": "dense", "units": 3}},
        ]
    }
    model = ModelBasic(id=1, model_name="scalar_shape_model", graph_ir=graph_ir)
    db_session.add(model)
    db_session.commit()

    response = client.get("/api/v1/model/architecture/1?include_stats=true")
    assert response.status_code == 200

    data = response.json()["data"]
    # Input(10) -> Dense(64): 10*64 + 64 = 704
    # Dense(64) -> Dense(32): 64*32 + 32 = 2080
    # Dense(32) -> Dense(3): 32*3 + 3 = 99
    # Total: 704 + 2080 + 99 = 2883
    assert data["param_count"] == 2883
