from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/c2si_db"
    secret_key: str = "change-me"
    cors_allowed_origin: str = "http://localhost:3300"
    upload_folder: str = "./data"
    max_content_length: int = 200 * 1024 * 1024
    api_base: str = "/api/v1"
    debug: bool = True

    model_config = {"env_file": ".env"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
