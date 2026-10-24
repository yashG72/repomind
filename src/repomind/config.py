"""Application settings, loaded from environment variables (and an optional .env file).

Every setting has a safe default so the app can start with no configuration at all.
Secrets (like the LLM API key) are typed as SecretStr so they never appear in logs or reprs.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REPOMIND_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "RepoMind"
    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "json"

    # Repository ingestion limits (enforced from Phase 2 onwards)
    max_file_size_bytes: int = Field(default=1_000_000, gt=0)
    max_repo_size_bytes: int = Field(default=200_000_000, gt=0)
    max_files_per_repo: int = Field(default=5_000, gt=0)

    # Embeddings and vector store (used from Phase 3 onwards)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "repomind_chunks"

    # LLM provider (used from Phase 4 onwards)
    llm_provider: str = "none"
    llm_model: str = ""
    llm_api_key: SecretStr | None = None

    # Optional GitHub token, only to raise API rate limits for public repositories
    github_token: SecretStr | None = None

    @field_validator("qdrant_api_key", "llm_api_key", "github_token", mode="before")
    @classmethod
    def _blank_secret_is_none(cls, value: object) -> object:
        # `REPOMIND_LLM_API_KEY=` (left blank, as in .env.example) means "not set".
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings. Cached so the environment is read once."""
    return Settings()
