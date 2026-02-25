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
import app.models  # noqa: E402, F401
from app.database import get_db  # noqa: E402
from app.main import app  # noqa: E402

_TEST_ENGINE = create_engine(
    "sqlite:///./test_tensormap.db",
    connect_args={"check_same_thread": False},
)

db_module.engine = _TEST_ENGINE


@pytest.fixture(name="client", scope="function")
def client_fixture():
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
