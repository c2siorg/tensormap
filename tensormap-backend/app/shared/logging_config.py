"""Centralized logging configuration for the backend."""

import logging
import os


def get_logger(name: str) -> logging.Logger:
    """Create or retrieve a named logger with a console handler at the configured level."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))

        handler = logging.StreamHandler()
        handler.setLevel(getattr(logging, level, logging.INFO))
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
