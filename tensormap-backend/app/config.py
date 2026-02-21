"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """App-wide settings populated from environment variables and .env file."""

    database_url: str
    secret_key: str
    cors_allowed_origins: str = "http://localhost:3300"
    upload_folder: str = "./data"
    max_content_length: int = 200 * 1024 * 1024
    api_base: str = "/api/v1"
    debug: bool = True

    model_config = {"env_file": ".env"}

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        """Parse the comma-separated CORS origins string into a list."""
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (created once per process)."""
    return Settings()
