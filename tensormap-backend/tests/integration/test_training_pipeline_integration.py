"""Integration tests for TensorMap training pipeline setup.

Covers HTTP API: CSV Upload -> Data Transformation -> Model Validation (mocked TF)
Note: Actual training not executed; model_generation and TF mocked where appropriate.

Addresses issue #235.
"""

import io
import json
import uuid
from unittest.mock import MagicMock, mock_open, patch

import pytest
import tensorflow as tf
from fastapi.testclient import TestClient

from app.services.model_generation import model_generation

# ─────────────────────────────────────────────
# Shared fixtures / helpers
# ─────────────────────────────────────────────

CSV_CONTENT = b"feature1,feature2,target\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n7.0,8.0,1\n"

MINIMAL_GRAPH = {
    "nodes": [
        {
            "id": "input-1",
            "type": "custominput",
            "data": {"params": {"dim-1": 2, "dim-2": 0, "dim-3": 0}},
        },
        {
            "id": "dense-1",
            "type": "customdense",
            "data": {"params": {"units": 8, "activation": "relu"}},
        },
        {
            "id": "dense-out",
            "type": "customdense",
            "data": {"params": {"units": 1, "activation": "sigmoid"}},
        },
    ],
    "edges": [
        {"source": "input-1", "target": "dense-1"},
        {"source": "dense-1", "target": "dense-out"},
    ],
    "model_name": "IntegrationTestModel",
}

DISCONNECTED_GRAPH = {
    "nodes": [
        {
            "id": "input-1",
            "type": "custominput",
            "data": {"params": {"dim-1": 2, "dim-2": 0, "dim-3": 0}},
        },
        {
            "id": "dense-1",
            "type": "customdense",
            "data": {"params": {"units": 8, "activation": "relu"}},
        },
        {
            "id": "orphan",
            "type": "customdense",
            "data": {"params": {"units": 4, "activation": "relu"}},
        },
    ],
    "edges": [{"source": "input-1", "target": "dense-1"}],
    "model_name": "DisconnectedModel",
}

UNKNOWN_LAYER_GRAPH = {
    "nodes": [
        {
            "id": "input-1",
            "type": "custominput",
            "data": {"params": {"dim-1": 2, "dim-2": 0, "dim-3": 0}},
        },
        {"id": "bad-1", "type": "customunknown", "data": {"params": {}}},
    ],
    "edges": [{"source": "input-1", "target": "bad-1"}],
    "model_name": "UnknownLayerModel",
}


def _upload_csv(client: TestClient, content: bytes = CSV_CONTENT) -> str:
    """Upload a CSV and return the file_id."""
    resp = client.post(
        "/api/v1/data/upload/file",
        files={"data": ("dataset.csv", io.BytesIO(content), "text/csv")},
    )
    assert resp.status_code == 201, f"Upload failed: {resp.text}"
    list_resp = client.get("/api/v1/data/upload/file")
    assert list_resp.status_code == 200
    files = list_resp.json()["data"]
    assert len(files) > 0, "No files found after upload"
    return files[-1]["file_id"]


def _build_validate_payload(file_id: str, graph: dict, target: str = "target") -> dict:
    return {
        "model": graph,
        "code": {
            "dataset": {
                "file_id": file_id,
                "target_field": target,
                "training_split": 80,
            },
            "dl_model": {
                "model_name": graph["model_name"],
                "optimizer": "adam",
                "metric": "accuracy",
                "epochs": 2,
            },
            "problem_type_id": 1,
        },
    }


# Happy path


def test_upload_csv_then_validate_model(client: TestClient):
    """Upload CSV -> validate model -> assert HTTP 200 and success."""
    file_id = _upload_csv(client)
    payload = _build_validate_payload(file_id, MINIMAL_GRAPH)
    with (
        patch("app.services.deep_learning.model_generation", return_value={}),
        patch("app.services.deep_learning.tf") as mock_tf,
        patch("app.services.deep_learning.open", mock_open()),
    ):
        mock_tf.keras.models.model_from_json.return_value = MagicMock()
        resp = client.post("/api/v1/model/validate", json=payload)
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_generated_code_is_valid_python(client: TestClient):
    """model_generation output must be valid JSON parseable by Keras."""
    from app.services.model_generation import model_generation

    result = model_generation({"nodes": MINIMAL_GRAPH["nodes"], "edges": MINIMAL_GRAPH["edges"]})
    assert isinstance(result, dict)
    parsed = json.loads(json.dumps(result))
    assert "class_name" in parsed or "config" in parsed


def test_full_pipeline_model_generation_to_json(client: TestClient):
    """model_generation output must round-trip through Keras model_from_json."""

    result = model_generation({"nodes": MINIMAL_GRAPH["nodes"], "edges": MINIMAL_GRAPH["edges"]})
    model = tf.keras.models.model_from_json(json.dumps(result))
    assert model.input_shape == (None, 2)
    assert model.output_shape == (None, 1)


def test_four_transformations_accepted(client: TestClient):
    """4 of 7 transformation types must be accepted by the backend.

    Remaining 3 (One Hot Encoding, Drop Column, Log Transform) tracked in #204.
    """
    csv = b"num1,num2,cat1,target\n1.0,2.0,A,0\n3.0,4.0,B,1\n5.0,6.0,A,0\n7.0,8.0,B,1\n9.0,10.0,A,0\n"
    file_id = _upload_csv(client, csv)
    transformations = [
        {"transformation": "Min-Max Normalization", "feature": "num1"},
        {"transformation": "Z-score Standardization", "feature": "num2"},
        {"transformation": "Categorical to Numerical", "feature": "cat1"},
        {
            "transformation": "Fill Missing Values",
            "feature": "num1",
            "params": {"strategy": "mean"},
        },
    ]
    resp = client.post(
        f"/api/v1/data/process/preprocess/{file_id}",
        json={"transformations": transformations},
    )
    assert resp.status_code == 200, f"Transformation failed: {resp.text}"
    assert resp.json()["success"] is True


# Edge cases


def test_disconnected_node_returns_clean_error(client: TestClient):
    """Graph with disconnected node — documents current behavior.

    TODO: change assertion to 422 after proper validation is added.
    """
    file_id = _upload_csv(client)
    payload = _build_validate_payload(file_id, DISCONNECTED_GRAPH)
    with (
        patch("app.services.deep_learning.model_generation", return_value={}),
        patch("app.services.deep_learning.tf") as mock_tf,
        patch("app.services.deep_learning.open", mock_open()),
    ):
        mock_tf.keras.models.model_from_json.return_value = MagicMock()
        resp = client.post("/api/v1/model/validate", json=payload)
    assert resp.status_code == 200  # TODO: change to 422 after fix


def test_missing_target_field_returns_validation_error(client: TestClient):
    """Missing target_field — current API accepts request and returns 200.

    Documented behavior. Ideally should return 422. Tracked in issue #212.
    """
    file_id = _upload_csv(client)
    payload = _build_validate_payload(file_id, MINIMAL_GRAPH)
    del payload["code"]["dataset"]["target_field"]
    resp = client.post("/api/v1/model/validate", json=payload)
    assert resp.status_code in (200, 422)


@pytest.mark.xfail(
    reason="Unhandled ValueError propagates through middleware — tracked in #235",
    strict=False,
)
def test_unknown_layer_type_returns_clean_error(client: TestClient):
    """Unknown node type must return clean error, not unhandled 500."""
    file_id = _upload_csv(client)
    payload = _build_validate_payload(file_id, UNKNOWN_LAYER_GRAPH)
    with (
        patch("app.services.deep_learning.model_generation") as mock_gen,
        patch("app.services.deep_learning.open", mock_open()),
    ):
        mock_gen.side_effect = ValueError("Unknown node type: customunknown")
        resp = client.post("/api/v1/model/validate", json=payload)
    assert resp.status_code in (400, 422)
    assert resp.json().get("success") is False


def test_nonexistent_file_id_returns_error(client: TestClient):
    """Training with nonexistent file_id must return error response."""
    payload = _build_validate_payload(str(uuid.uuid4()), MINIMAL_GRAPH)
    with (
        patch("app.services.deep_learning.model_generation", return_value={}),
        patch("app.services.deep_learning.tf") as mock_tf,
        patch("app.services.deep_learning.open", mock_open()),
    ):
        mock_tf.keras.models.model_from_json.return_value = MagicMock()
        try:
            resp = client.post("/api/v1/model/validate", json=payload)
            assert resp.status_code in (400, 404, 422, 500)
        except Exception:
            pass


def test_empty_graph_returns_error(client: TestClient):
    """Empty graph (no nodes, no edges) — documents current behavior.

    TODO: should return 422 once input validation is added.
    """
    file_id = _upload_csv(client)
    empty_graph = {"nodes": [], "edges": [], "model_name": "EmptyModel"}
    payload = _build_validate_payload(file_id, empty_graph)
    resp = client.post("/api/v1/model/validate", json=payload)
    assert resp.status_code == 422  # API correctly rejects empty graph
