"""Unit tests for the health and readiness check endpoints."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.database import get_db
from app.main import app

client = TestClient(app)


def test_health_check_returns_200():
    """Liveness probe returns 200 OK with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_check_success():
    """Readiness probe returns 200 OK when database is connected."""
    mock_db = MagicMock()
    mock_db.exec.return_value = None

    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready", "database": "connected"}
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_readiness_check_failure_returns_503():
    """Readiness probe returns 503 Service Unavailable when database connection fails."""
    mock_db = MagicMock()
    mock_db.exec.side_effect = Exception("Database connection refused")

    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json() == {"status": "not ready", "database": "disconnected"}
    finally:
        app.dependency_overrides.pop(get_db, None)
