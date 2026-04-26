"""Application configuration loaded from environment variables."""

from functools import lru_cache
from urllib.parse import urlparse

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """App-wide settings populated from environment variables and .env file."""

    database_url: str
    secret_key: str
    cors_allowed_origins: str = "http://localhost:3300"
    upload_folder: str = "./data"
    max_content_length: int = 200 * 1024 * 1024
    api_base: str = "/api/v1"
    debug: bool = False

    # Ignore unrelated environment variables injected by CI/container runtimes.
    model_config = {"env_file": ".env", "extra": "ignore"}

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Ensure the database URL uses a recognised scheme."""
        # postgresql+asyncpg removed — engine is synchronous (sqlmodel.create_engine)
        allowed_schemes = ("postgresql", "postgresql+psycopg2", "sqlite")
        if not any(v.startswith(scheme) for scheme in allowed_schemes):
            raise ValueError(f"database_url must start with one of {allowed_schemes}. Got: {v!r}")
        return v

    @field_validator("cors_allowed_origins")
    @classmethod
    def validate_cors_origins(cls, v: str) -> str:
        """Ensure every CORS origin is a valid URL (or the wildcard '*')."""
        origins = [o.strip() for o in v.split(",") if o.strip()]
        for origin in origins:
            if origin == "*":
                continue
            parsed = urlparse(origin)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError(
                    f"Invalid CORS origin {origin!r}. Must be a full URL (e.g. 'https://example.com') or '*'."
                )
        return v

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        """Parse the comma-separated CORS origins string into a list."""
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (created once per process)."""
    return Settings()
