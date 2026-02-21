"""TensorMap backend application entry point.

Creates the FastAPI app, configures CORS and exception handlers, mounts
routers, and wraps the ASGI app with Socket.IO for real-time training progress.
"""

from contextlib import asynccontextmanager

import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.exceptions import AppException, app_exception_handler, generic_exception_handler
from app.routers import data_process, data_upload, deep_learning, project
from app.shared.logging_config import get_logger
from app.socketio_instance import sio

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run Alembic migrations on startup, then yield control to the app."""
    from alembic import command
    from alembic.config import Config

    logger.info("Running Alembic migrations...")
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
    logger.info("Alembic migrations complete")
    yield


app = FastAPI(title="TensorMap API", lifespan=lifespan)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

app.include_router(data_upload.router, prefix=settings.api_base)
app.include_router(data_process.router, prefix=settings.api_base)
app.include_router(deep_learning.router, prefix=settings.api_base)
app.include_router(project.router, prefix=settings.api_base)

# Wrap FastAPI with SocketIO so socket.io requests are handled,
# and everything else passes through to FastAPI.
# Run this combined app with: uvicorn app.main:socket_app
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)
