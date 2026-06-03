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
    # Use a unique model name to avoid conflicts
    simple_graph["model_name"] = "test_save_valid"
    simple_graph["model"]["model_name"] = "test_save_valid"
    response = client.post("/api/v1/model/save", json=simple_graph)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True


def test_model_list_after_save(client, simple_graph):
    """After saving a model it should appear in the model list."""
    # Use a unique model name to avoid conflicts
    simple_graph["model_name"] = "test_list_after"
    simple_graph["model"]["model_name"] = "test_list_after"
    
    # First save the model
    save_response = client.post("/api/v1/model/save", json=simple_graph)
    assert save_response.status_code == 200
    
    # Then check if it appears in the list
    response = client.get("/api/v1/model/model-list")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    model_names = [m["model_name"] for m in body["data"]]
    assert "test_list_after" in model_names
