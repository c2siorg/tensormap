"""Tests for RequestLoggingMiddleware — verifies every request is logged."""

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.middleware import RequestLoggingMiddleware


@pytest.fixture()
def client():
    """Minimal FastAPI app with only the logging middleware — no DB, no migrations."""
    test_app = FastAPI()
    test_app.add_middleware(RequestLoggingMiddleware)

    @test_app.get("/health")
    def health():
        return {"ok": True}

    @test_app.post("/echo")
    def echo():
        return {"ok": True}

    @test_app.get("/not-found")
    def not_found():
        return JSONResponse(status_code=404, content={"ok": False})

    return TestClient(test_app)


def test_get_request_is_logged(client, caplog):
    """GET /health → 200 must appear in the log."""
    with caplog.at_level("INFO", logger="app.middleware"):
        client.get("/health")
    assert "GET" in caplog.text
    assert "/health" in caplog.text
    assert "200" in caplog.text


def test_post_request_is_logged(client, caplog):
    """POST /echo → 200 must appear in the log."""
    with caplog.at_level("INFO", logger="app.middleware"):
        client.post("/echo")
    assert "POST" in caplog.text
    assert "/echo" in caplog.text
    assert "200" in caplog.text


def test_duration_ms_is_logged(client, caplog):
    """Log line must contain a duration ending in 'ms'."""
    with caplog.at_level("INFO", logger="app.middleware"):
        client.get("/health")
    assert "ms" in caplog.text


def test_non_200_status_is_logged(client, caplog):
    """404 responses must be logged with the correct status code."""
    with caplog.at_level("INFO", logger="app.middleware"):
        client.get("/not-found")
    assert "404" in caplog.text


def test_existing_handlers_unaffected(client):
    """Sanity: middleware does not corrupt the response body."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
