import json
import pytest

# ── 1. Dataset Upload ──────────────────────────────────────────────
def test_dataset_upload(client, sample_csv):
    response = client.post(
        "/data/upload",
        data={"file": (sample_csv, "sample_dataset.csv")},
        content_type="multipart/form-data"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "file_id" in data or "id" in data  # adjust to actual response key

# ── 2. Code Generation from Graph ─────────────────────────────────
def test_graph_to_code_generation(client, simple_graph):
    response = client.post(
        "/model/generate-code",
        json=simple_graph,
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "code" in data
    assert "model.compile" in data["code"] or "Sequential" in data["code"]

# ── 3. Invalid Graph Rejected Cleanly ─────────────────────────────
def test_invalid_graph_rejected(client):
    bad_graph = {"nodes": [], "edges": []}  # empty = invalid
    response = client.post(
        "/model/generate-code",
        json=bad_graph,
        content_type="application/json"
    )
    assert response.status_code in [400, 422]
    data = response.get_json()
    assert "error" in data or "message" in data

# ── 4. Missing Target Field Returns Clean Error ────────────────────
def test_missing_target_field_error(client):
    response = client.post(
        "/model/train",
        json={"file_id": 999, "model_name": "test_model"},  # no target_field
        content_type="application/json"
    )
    assert response.status_code in [400, 422]
