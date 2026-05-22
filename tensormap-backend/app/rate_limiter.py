"""Rate limiting utilities for API endpoints."""

import time
from collections import defaultdict

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.shared.logging_config import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Token bucket rate limiter keyed by client IP."""

    def __init__(self, requests_per_minute: int = 60, window_seconds: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self._clients: dict[str, list[float]] = defaultdict(list)

    def _get_client_id(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        client_host = request.client.host if request.client else "unknown"
        return client_host

    def check(self, request: Request) -> None:
        """Raise HTTPException 429 if the client has exceeded the rate limit."""
        client_id = self._get_client_id(request)
        now = time.time()
        window_start = now - self.window_seconds

        timestamps = self._clients[client_id]
        timestamps[:] = [t for t in timestamps if t > window_start]

        if len(timestamps) >= self.requests_per_minute:
            logger.warning("Rate limit exceeded for client %s", client_id)
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please slow down.")

        timestamps.append(now)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware that applies rate limiting to all incoming requests."""

    def __init__(self, app, requests_per_minute: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            window_seconds=window_seconds,
        )

    async def dispatch(self, request: Request, call_next):
        self.limiter.check(request)
        response = await call_next(request)
        return response
