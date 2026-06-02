"""Tests for the rate limiter utility."""

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from app.rate_limiter import RateLimiter, RateLimitExceeded, RateLimitMiddleware


class TestRateLimiter:
    def test_allows_requests_under_limit(self):
        """Requests within the limit should be allowed."""
        limiter = RateLimiter(requests_per_minute=5, window_seconds=60)
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.client.host = "1.2.3.4"

        for _ in range(5):
            limiter.check(mock_request)

    def test_blocks_requests_over_limit(self):
        """Requests exceeding the limit should raise 429."""
        limiter = RateLimiter(requests_per_minute=3, window_seconds=60)
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.client.host = "1.2.3.4"

        for _ in range(3):
            limiter.check(mock_request)

        with pytest.raises(RateLimitExceeded):
            limiter.check(mock_request)

    def test_allows_requests_after_window_expires(self):
        """After the window expires, requests should be allowed again."""
        limiter = RateLimiter(requests_per_minute=2, window_seconds=60)
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.client.host = "1.2.3.4"

        limiter.check(mock_request)
        limiter.check(mock_request)

        with pytest.raises(RateLimitExceeded):
            limiter.check(mock_request)

        future = time.time() + 120
        with patch.object(time, "time", return_value=future):
            limiter.check(mock_request)

    def test_uses_forwarded_for_header(self):
        """If X-Forwarded-For is present from a trusted proxy, it should be used as the client ID."""
        limiter = RateLimiter(requests_per_minute=1, window_seconds=60, trusted_proxies={"1.2.3.4"})
        mock_request = MagicMock()
        mock_request.headers = {"X-Forwarded-For": "5.6.7.8, 9.10.11.12"}
        mock_request.client.host = "1.2.3.4"

        limiter.check(mock_request)
        # Second request from the forwarded IP should be blocked (separate from proxy's own rate limit)
        with pytest.raises(RateLimitExceeded):
            limiter.check(mock_request)

    def test_different_clients_have_separate_limits(self):
        """Two different clients should not share rate limit state."""
        limiter = RateLimiter(requests_per_minute=2, window_seconds=60)

        client_a = MagicMock()
        client_a.headers = {}
        client_a.client.host = "1.1.1.1"

        client_b = MagicMock()
        client_b.headers = {}
        client_b.client.host = "2.2.2.2"

        limiter.check(client_a)
        limiter.check(client_a)

        with pytest.raises(RateLimitExceeded):
            limiter.check(client_a)

        limiter.check(client_b)
        limiter.check(client_b)


class TestRateLimitMiddleware:
    def test_middleware_allows_normal_requests(self):
        """Normal requests within the limit should succeed."""
        app = Starlette(
            routes=[
                Route("/health", lambda r: JSONResponse({"ok": True})),
            ],
        )
        app.add_middleware(RateLimitMiddleware, requests_per_minute=100, window_seconds=60)
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200

    def test_middleware_blocks_excessive_requests(self):
        """Requests exceeding the limit should get 429."""
        app = Starlette(
            routes=[
                Route("/health", lambda r: JSONResponse({"ok": True})),
            ],
        )
        app.add_middleware(RateLimitMiddleware, requests_per_minute=2, window_seconds=60)
        client = TestClient(app)

        client.get("/health")
        client.get("/health")

        response = client.get("/health")
        assert response.status_code == 429
