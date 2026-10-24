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

    # Repository ingestion limits
    # Largest single file that is indexed; bigger files are skipped.
    max_file_size_bytes: int = Field(default=1_000_000, gt=0)
    # Largest total uncompressed repository content; bigger repositories are refused.
    max_repo_size_bytes: int = Field(default=200_000_000, gt=0)
    # Largest compressed archive download; the download is aborted past this point.
    max_download_bytes: int = Field(default=100_000_000, gt=0)
    # At most this many files are indexed; the rest are reported as skipped.
    max_files_per_repo: int = Field(default=5_000, gt=0)
    # Timeout for each request to GitHub.
    github_timeout_seconds: float = Field(default=30.0, gt=0)

    # Chunking
    chunk_max_lines: int = Field(default=80, ge=5)
    chunk_max_chars: int = Field(default=6_000, ge=200)

    # Embeddings
    # A Hugging Face model id, or a path to a local Sentence Transformers model folder.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=32, ge=1, le=1024)
    # Where downloaded models are cached (default: the Hugging Face cache in your home folder).
    embedding_cache_dir: str | None = None
    # true = never download; the model must already be cached or be a local path.
    embedding_local_files_only: bool = False

    # Qdrant vector store
    # Set qdrant_url to use a Qdrant server (Docker or Qdrant Cloud). Leave it empty to use
    # embedded local storage in qdrant_path, which needs no server.
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_path: str = "data/qdrant"
    qdrant_collection: str = "repomind_chunks"
    qdrant_timeout_seconds: int = Field(default=10, ge=1, le=300)
    upsert_batch_size: int = Field(default=128, ge=1, le=4096)

    # Search
    search_default_top_k: int = Field(default=5, ge=1, le=50)
    search_max_query_chars: int = Field(default=2_000, ge=10, le=20_000)

    # LLM provider (used from Phase 4 onwards)
    llm_provider: str = "none"
    llm_model: str = ""
    llm_api_key: SecretStr | None = None

    # Optional GitHub token, only to raise API rate limits for public repositories
    github_token: SecretStr | None = None

    @field_validator(
        "qdrant_api_key",
        "llm_api_key",
        "github_token",
        "qdrant_url",
        "embedding_cache_dir",
        mode="before",
    )
    @classmethod
    def _blank_means_unset(cls, value: object) -> object:
        # `REPOMIND_LLM_API_KEY=` (left blank, as in .env.example) means "not set".
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings. Cached so the environment is read once."""
    return Settings()
