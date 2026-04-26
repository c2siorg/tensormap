"""Retry utilities for transient failures."""

import time
from functools import wraps

from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry decorator for functions that may fail transiently."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == max_attempts:
                        logger.error("%s failed after %d attempts", func.__name__, max_attempts)
                        raise
                    logger.warning("%s failed (attempt %d), retrying", func.__name__, attempt)
                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper

    return decorator
