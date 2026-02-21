"""Shared Socket.IO server instance used for real-time training progress."""

import socketio

from app.config import get_settings
from app.shared.constants import SOCKETIO_DL_NAMESPACE

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=get_settings().cors_allowed_origins_list,
)


@sio.on("connect", namespace=SOCKETIO_DL_NAMESPACE)
async def dl_connect(sid, environ):
    """Accept client connections to the training progress namespace."""
    pass
