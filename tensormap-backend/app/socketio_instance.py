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
    if not sid:
        logger.warning("Rejected Socket.IO client with empty session id")
        return False

    client_ip = environ.get("REMOTE_ADDR", "unknown")
    origin = environ.get("HTTP_ORIGIN", "unknown")
    logger.info("Socket.IO client connected: sid=%s ip=%s origin=%s", sid, client_ip, origin)
    return True
