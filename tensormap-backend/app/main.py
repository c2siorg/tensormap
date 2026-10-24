"""TensorMap backend application entry point.
Creates the FastAPI app, configures CORS and exception handlers, mounts
routers, and wraps the ASGI app with Socket.IO for real-time training progress.
"""

import os
from contextlib import asynccontextmanager

import socketio
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from app.config import get_settings
from app.exceptions import (
    AppException,
    app_exception_handler,
    generic_exception_handler,
    integrity_error_handler,
    validation_exception_handler,
)
from app.middleware import RequestIDMiddleware, RequestLoggingMiddleware
from app.routers import (
    analysis,
    data_process,
    data_upload,
    deep_learning,
    export,
    health,
    layers,
    project,
    training,
    tuning,
)
from app.shared.logging_config import get_logger
from app.socketio_instance import sio

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run Alembic migrations on startup, then yield control to the app."""
    import asyncio

    from alembic import command
    from alembic.config import Config
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory

    cleanup_task = None

    if os.environ.get("TESTING"):
        logger.info("TESTING mode — skipping Alembic migrations")
    else:
        logger.info("Checking database migrations...")
        try:
            from app.database import engine

            alembic_cfg = Config("alembic.ini")
            script = ScriptDirectory.from_config(alembic_cfg)

            with engine.connect() as connection:
                context = MigrationContext.configure(connection)
                current_rev = context.get_current_revision()
                head_rev = script.get_current_head()

                if current_rev == head_rev:
                    logger.info(f"Database already at head revision: {current_rev}")
                else:
                    logger.info(f"Upgrading from {current_rev} to {head_rev}...")
                    command.upgrade(alembic_cfg, "head")
                    logger.info("Alembic migrations complete")
        except Exception as e:
            logger.error(f"Migration error: {e}")
            logger.info("Attempting full migration upgrade...")
            alembic_cfg = Config("alembic.ini")
            command.upgrade(alembic_cfg, "head")
            logger.info("Alembic migrations complete")

        # Recover jobs left RUNNING/PENDING by a previous crash so the DB never
        # shows a job as active when no process is training it. Best-effort:
        # a failure here must not prevent the app from starting.
        try:
            from app.services.training_service import orphan_recovery
            from app.services.tuning_service import orphan_recovery_tuning

            orphan_recovery()
            orphan_recovery_tuning()
        except Exception as e:
            logger.error(f"Orphan recovery error: {e}")

        # Start background cleanup task for old export files
        async def _export_cleanup_loop():
            from app.database import get_session
            from app.services.model_export import EXPORTS_BASE, cleanup_exports

            while True:
                await asyncio.sleep(3600)  # hourly
                try:
                    retention_days = int(os.getenv("EXPORT_RETENTION_DAYS", "7"))
                    max_storage_gb = float(os.getenv("MAX_EXPORT_STORAGE_GB", "5.0"))

                    # Check total storage usage
                    if EXPORTS_BASE.exists():
                        total_bytes = sum(f.stat().st_size for f in EXPORTS_BASE.rglob("*") if f.is_file())
                        total_gb = total_bytes / (1024**3)
                        logger.debug(f"Export storage usage: {total_gb:.2f} GB / {max_storage_gb} GB")

                        # If over limit, use aggressive cleanup (1 day retention)
                        if total_gb > max_storage_gb:
                            logger.warning(f"Export storage limit exceeded ({total_gb:.2f} GB > {max_storage_gb} GB)")
                            with get_session() as session:
                                count = cleanup_exports(1, session)  # aggressive cleanup
                            logger.info(f"Aggressive cleanup: deleted {count} directories")
                        else:
                            # Normal cleanup with configured retention
                            with get_session() as session:
                                count = cleanup_exports(retention_days, session)
                            if count > 0:
                                logger.info(f"Export cleanup: deleted {count} directories")
                except Exception as e:
                    logger.error(f"Export cleanup error: {e}")

        cleanup_task = asyncio.create_task(_export_cleanup_loop())

    yield

    # Cleanup on shutdown
    if cleanup_task:
        import contextlib

        cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cleanup_task


app = FastAPI(title="TensorMap API", lifespan=lifespan)
settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RequestIDMiddleware)

app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(IntegrityError, integrity_error_handler)
app.add_exception_handler(ValidationError, validation_exception_handler)
# FastAPI raises RequestValidationError for invalid request bodies/params,
# not pydantic.ValidationError — register this to standardise the response shape.
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

app.include_router(health.router, tags=["health"])
app.include_router(data_upload.router, prefix=settings.api_base)
app.include_router(data_process.router, prefix=settings.api_base)
app.include_router(deep_learning.router, prefix=settings.api_base)
app.include_router(project.router, prefix=settings.api_base)
app.include_router(layers.router, prefix=settings.api_base)
app.include_router(training.router, prefix=settings.api_base)
app.include_router(export.router, prefix=settings.api_base)
app.include_router(analysis.router, prefix=settings.api_base)
app.include_router(tuning.router, prefix=settings.api_base)

# Wrap FastAPI with SocketIO so socket.io requests are handled,
# and everything else passes through to FastAPI.
# Run this combined app with: uvicorn app.main:socket_app
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)
