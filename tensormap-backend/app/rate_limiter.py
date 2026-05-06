"""Rate limiting utilities for API endpoints."""

import time
from collections import defaultdict

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.shared.logging_config import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_minute: int = 10, window_seconds: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self._clients: dict[str, list[float]] = defaultdict(list)

    def _get_client_id(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _cleanup_old_requests(self, client_id: str) -> None:
        cutoff = time.time() - self.window_seconds
        self._clients[client_id] = [ts for ts in self._clients[client_id] if ts > cutoff]

    def check(self, request: Request) -> bool:
        client_id = self._get_client_id(request)
        self._cleanup_old_requests(client_id)
        if len(self._clients[client_id]) >= self.requests_per_minute:
            return False
        self._clients[client_id].append(time.time())
        return True

    def get_remaining(self, request: Request) -> int:
        client_id = self._get_client_id(request)
        self._cleanup_old_requests(client_id)
        return max(0, self.requests_per_minute - len(self._clients[client_id]))


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce rate limits on specified paths."""

    def __init__(self, app, limiter: RateLimiter, paths: list[str]):
        super().__init__(app)
        self.limiter = limiter
        self.paths = paths

    async def dispatch(self, request: Request, call_next):
        if not any(request.url.path.startswith(p) for p in self.paths):
            return await call_next(request)

        if not self.limiter.check(request):
            raise HTTPException(status_code=429, detail="Too many requests")

        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(self.limiter.get_remaining(request))
        return response


def get_training_limiter() -> RateLimiter:
    """Get rate limiter for training endpoints."""
    return RateLimiter(requests_per_minute=5, window_seconds=60)
