import json
from unittest.mock import patch, mock_open
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_get_layer_registry_success():
    """Test that a valid registry file is returned correctly."""
    mock_data = json.dumps({"layers": {"dense": {"display_name": "Dense"}}})
    
    with patch("app.routers.deep_learning._LAYER_REGISTRY", json.loads(mock_data)):
        response = client.get("/api/v1/layers")
        
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert "dense" in response.json()["data"]["layers"]

def test_get_layer_registry_unavailable():
    """Test that the endpoint returns a 500 if the registry failed to load on startup."""
    with patch("app.routers.deep_learning._LAYER_REGISTRY", None):
        response = client.get("/api/v1/layers")
        
    assert response.status_code == 500
    assert response.json()["success"] is False

