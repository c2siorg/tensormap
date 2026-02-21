"""Custom exception classes and global exception handlers for the API."""

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from app.shared.logging_config import get_logger

logger = get_logger(__name__)


class AppException(HTTPException):
    """Application-level HTTP exception with a user-facing message."""

    def __init__(self, status_code: int, message: str):
        super().__init__(status_code=status_code, detail=message)
        self.message = message


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle AppException and return a standard JSON error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "message": exc.message, "data": None},
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unhandled exceptions, returning a 500 response."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": "Internal server error", "data": None},
    )
