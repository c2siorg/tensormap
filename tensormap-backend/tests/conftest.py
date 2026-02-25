"""Shared pytest fixtures for API integration tests.

os.environ assignments at the module top MUST happen before any app import so that
get_settings() (lru_cached) and database.engine (module-level create_engine call)
both see the SQLite URL and temp upload folder.
"""
import os
import tempfile

_upload_dir = tempfile.mkdtemp(prefix="tensormap_test_")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_tensormap.db")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("UPLOAD_FOLDER", _upload_dir)

from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlmodel import SQLModel, Session, create_engine  # noqa: E402

import app.database as db_module  # noqa: E402
import app.models  # noqa: E402, F401 — registers all SQLModel table metadata
from app.database import get_db  # noqa: E402
from app.main import app  # noqa: E402

# Create a SQLite engine with check_same_thread=False (required because
# TestClient runs the ASGI app in a separate thread).
_TEST_ENGINE = create_engine(
    "sqlite:///./test_tensormap.db",
    connect_args={"check_same_thread": False},
)

# Replace the module-level engine so any service code that references
# app.database.engine directly also uses the test database.
db_module.engine = _TEST_ENGINE


@pytest.fixture(name="client", scope="function")
def client_fixture():
    """TestClient backed by an isolated SQLite DB — schema recreated per test (function scope)."""
    SQLModel.metadata.drop_all(_TEST_ENGINE)
    SQLModel.metadata.create_all(_TEST_ENGINE)

    def override_get_db():
        with Session(_TEST_ENGINE) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    with patch("alembic.command.upgrade"):
        with TestClient(app) as c:
            yield c

    app.dependency_overrides.pop(get_db, None)
