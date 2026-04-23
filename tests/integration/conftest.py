"""Shared fixtures for integration tests."""

import io
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

# Must be set before any app imports
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["SECRET_KEY"] = "integration-test-secret-key-not-for-production"


def _make_engine():
    import app.models  # noqa: F401 — registers all SQLModel tables

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture(scope="session")
def engine():
    return _make_engine()


@pytest.fixture(scope="function")
def db_session(engine):
    """Function-scoped session — each test gets a clean transaction."""
    with Session(engine) as session:
        yield session
        session.rollback()


@pytest.fixture(scope="session")
def client(engine):
    """Session-scoped TestClient with Alembic migrations patched out."""
    with patch("alembic.command.upgrade"):
        from app.database import get_db
        from app.main import app

        def get_test_db():
            with Session(engine) as session:
                yield session

        app.dependency_overrides[get_db] = get_test_db

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

        app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def sample_csv_bytes():
    return b"feature1,feature2,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n9,10,0\n"


@pytest.fixture(scope="session")
def simple_graph():
    return {
        "model": {
            "nodes": [
                {
                    "id": "input1",
                    "type": "custominput",
                    "data": {"params": {"dim-1": 2, "dim-2": "", "dim-3": ""}},
                    "position": {"x": 0, "y": 0},
                },
                {
                    "id": "dense1",
                    "type": "customdense",
                    "data": {"params": {"units": 4, "activation": "relu"}},
                    "position": {"x": 0, "y": 200},
                },
                {
                    "id": "dense2",
                    "type": "customdense",
                    "data": {"params": {"units": 1, "activation": "sigmoid"}},
                    "position": {"x": 0, "y": 400},
                },
            ],
            "edges": [
                {"source": "input1", "target": "dense1"},
                {"source": "dense1", "target": "dense2"},
            ],
            "model_name": "test_model",
        },
        "model_name": "test_model",
    }
