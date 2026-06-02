"""HTTP request/response logging and tracing middleware."""

import logging
import re
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.shared.logging_config import get_logger

logger = get_logger(__name__)

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
_INT_SEGMENT_RE = re.compile(r"(?<=/)\d+(?=/|$)")

request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class RequestIDFilter(logging.Filter):
    """Inject the current request ID into every log record as ``request_id``."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


# Register the filter on the root logger so every module benefits.
logging.getLogger().addFilter(RequestIDFilter())


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique X-Request-ID header to every response for request tracing."""

    async def dispatch(self, request: Request, call_next) -> Response:
        raw_id = request.headers.get("X-Request-ID", "")
        request_id = raw_id if raw_id and _UUID_RE.fullmatch(raw_id) else str(uuid.uuid4())

        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_var.reset(token)


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
            req_id = request_id_var.get()
            logger.info(
                "%s %s → %d (%.1fms) [request_id=%s]",
                request.method,
                path,
                status,
                duration_ms,
                req_id,
            )
