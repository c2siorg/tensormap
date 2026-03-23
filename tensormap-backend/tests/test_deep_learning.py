from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_get_layer_registry_success(client):
    """Test that a valid registry file is returned correctly."""
    mock_data = {"layers": {"dense": {"display_name": "Dense"}}}

    with patch("app.routers.deep_learning._LAYER_REGISTRY", mock_data):
        response = client.get("/api/v1/layers")

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert "dense" in response.json()["data"]["layers"]


def test_get_layer_registry_unavailable(client):
    """Test that the endpoint returns a 500 if the registry failed to load on startup."""
    with patch("app.routers.deep_learning._LAYER_REGISTRY", None):
        response = client.get("/api/v1/layers")

    assert response.status_code == 500
    assert response.json()["success"] is False


def test_get_layer_registry_contains_required_layers(client):
    response = client.get("/api/v1/layers")
    assert response.status_code == 200
    layers = response.json()["data"]["layers"]
    for expected in ["input", "dense", "flatten", "conv2d"]:
        assert expected in layers, f"Layer '{expected}' missing from registry"
