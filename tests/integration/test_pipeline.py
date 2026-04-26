"""End-to-end integration tests for the ML pipeline."""


def test_health_check(client):
    """Model list endpoint responds with success."""
    response = client.get("/api/v1/model/model-list")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "data" in body


def test_invalid_graph_rejected_empty_nodes(client):
    """Empty nodes list must be rejected with 422 and a validation error."""
    response = client.post(
        "/api/v1/model/save",
        json={
            "model": {"nodes": [], "edges": [], "model_name": "x"},
            "model_name": "x",
        },
    )
    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert "message" in body


def test_model_save_requires_model_name(client):
    """Missing model_name field must fail schema validation."""
    response = client.post(
        "/api/v1/model/save",
        json={"model": {"nodes": [], "edges": []}},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert "message" in body


def test_model_save_valid_graph(client, simple_graph):
    """A valid graph with proper nodes should save successfully."""
    response = client.post("/api/v1/model/save", json=simple_graph)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True


def test_model_list_after_save(client, simple_graph):
    """After saving a model it should appear in the model list."""
    response = client.get("/api/v1/model/model-list")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    model_names = [m["model_name"] for m in body["data"]]
    assert "test_model" in model_names
