"""Shared Socket.IO server instance used for real-time training progress."""

import socketio

from app.config import get_settings
from app.shared.constants import SOCKETIO_DL_NAMESPACE
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=get_settings().cors_allowed_origins_list,
)


@sio.on("connect", namespace=SOCKETIO_DL_NAMESPACE)
async def dl_connect(sid, environ):
    """Accept and log client connections to the training progress namespace."""
    client_ip = environ.get("REMOTE_ADDR", "unknown")
    logger.info("Client connected to training namespace: sid=%s, ip=%s", sid, client_ip)
