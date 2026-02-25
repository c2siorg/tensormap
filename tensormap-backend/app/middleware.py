"""HTTP request/response logging middleware."""

import re
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.shared.logging_config import get_logger

logger = get_logger(__name__)

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)
_INT_SEGMENT_RE = re.compile(r"(?<=/)\d+(?=/|$)")


def _sanitize_path(path: str) -> str:
    """Replace UUID and numeric path segments with {id} to avoid logging PII."""
    path = _UUID_RE.sub("{id}", path)
    path = _INT_SEGMENT_RE.sub("{id}", path)
    return path


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log method, path, status code, and duration for every HTTP request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            route = request.scope.get("route")
            path = route.path if route is not None else _sanitize_path(request.url.path)
            status = response.status_code if response is not None else 500
            logger.info(
                "%s %s → %d (%.1fms)",
                request.method,
                path,
                status,
                duration_ms,
            )
