import socketio

from app.config import get_settings

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=[get_settings().cors_allowed_origin],
)
