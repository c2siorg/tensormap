"""Rate limiting utilities for API endpoints."""

import time
from collections import defaultdict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.shared.logging_config import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Sliding-window rate limiter keyed by client IP."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        window_seconds: int = 60,
        trusted_proxies: set | None = None,
    ):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self.trusted_proxies = trusted_proxies or set()
        self._clients: dict[str, list[float]] = defaultdict(list)
        self._check_count = 0

    def _get_client_id(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded and self.trusted_proxies:
            if request.client and request.client.host in self.trusted_proxies:
                return forwarded.split(",")[0].strip()
        client_host = request.client.host if request.client else "unknown"
        return client_host

    def _prune_stale_keys(self) -> None:
        """Remove entries for clients with no recent activity."""
        now = time.time()
        cutoff = now - self.window_seconds * 2
        stale = [cid for cid, stamps in self._clients.items() if not stamps or stamps[-1] < cutoff]
        for cid in stale:
            del self._clients[cid]

    def check(self, request: Request) -> None:
        """Raise HTTPException 429 if the client has exceeded the rate limit."""
        client_id = self._get_client_id(request)
        now = time.time()
        window_start = now - self.window_seconds

        timestamps = self._clients[client_id]
        timestamps[:] = [t for t in timestamps if t > window_start]

        if len(timestamps) >= self.requests_per_minute:
            logger.warning("Rate limit exceeded for client %s", client_id)
            raise RateLimitExceeded()

        timestamps.append(now)
        self._check_count += 1
        if self._check_count % 100 == 0:
            self._prune_stale_keys()


class RateLimitExceeded(Exception):
    """Raised when a client exceeds the rate limit."""


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware that applies rate limiting to all incoming requests."""

    def __init__(self, app, requests_per_minute: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            window_seconds=window_seconds,
        )

    async def dispatch(self, request: Request, call_next):
        try:
            self.limiter.check(request)
        except RateLimitExceeded:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please slow down."},
            )
        response = await call_next(request)
        return response
