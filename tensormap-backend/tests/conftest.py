"""
Pytest configuration and fixtures for TensorMap backend tests.

This module provides test fixtures for database sessions, HTTP clients,
and test environment configuration with support for both PostgreSQL (CI)
and SQLite (local development) backends.
"""

import os

# Set TESTING flag *before* importing the app so the lifespan skips Alembic
# migrations — test fixtures manage the schema via SQLModel.metadata.create_all.
os.environ["TESTING"] = "1"

# Set required environment variables for Settings validation if not already set
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "sqlite:///./test.db"
if "SECRET_KEY" not in os.environ:
    os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only"

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from app.database import get_db
from app.main import app


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Configure anyio to use asyncio backend for async tests."""
    return "asyncio"


@pytest.fixture(scope="session")
def test_db_url() -> str:
    """
    Get database URL for tests.

    Returns PostgreSQL URL from DATABASE_URL env var if set (for CI),
    otherwise falls back to in-memory SQLite for local development.
    """
    return os.getenv("DATABASE_URL", "sqlite:///./test.db")


@pytest.fixture(scope="function")
def db_session(test_db_url: str) -> Generator[Session, None, None]:
    """
    Provide a clean database session for each test.

    Creates all tables before the test and drops them after,
    ensuring test isolation.

    Args:
        test_db_url: Database connection URL from test_db_url fixture

    Yields:
        Session: SQLModel database session
    """
    # Create engine with appropriate settings for SQLite vs PostgreSQL
    if test_db_url.startswith("sqlite"):
        engine = create_engine(
            test_db_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    else:
        engine = create_engine(test_db_url, echo=False)

    # Create all tables
    SQLModel.metadata.create_all(engine)

    # Provide session to test
    with Session(engine) as session:
        yield session

    # Clean up: drop all tables after test
    SQLModel.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="function")
def training_session_factory(monkeypatch) -> Generator:
    """A session factory over an isolated in-memory DB for training callbacks.

    Training callbacks and background workers open their own sessions via
    ``training_service.make_session`` (which reads the module-level engine).
    This fixture points that engine at a fresh in-memory SQLite DB so callback
    and service tests share one database without touching the app engine.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr("app.services.training_service.engine", engine)

    def factory() -> Session:
        return Session(engine)

    yield factory
    engine.dispose()


@pytest.fixture(scope="function")
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """
    Provide a test HTTP client for testing FastAPI endpoints.

    Overrides the app's database session dependency with the test session
    to ensure tests use the isolated test database.

    Args:
        db_session: Test database session from db_session fixture

    Yields:
        TestClient: FastAPI test client configured for the app
    """

    def override_get_db() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    # Clean up dependency override
    app.dependency_overrides.clear()
