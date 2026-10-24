"""Builds pipeline components from settings. Tests replace these with FastAPI's
dependency_overrides, for example to inject a fake GitHub transport."""

from typing import Annotated

from fastapi import Depends

from repomind.config import Settings, get_settings
from repomind.ingestion.chunking import CodeChunker
from repomind.ingestion.file_filter import DefaultFileFilter
from repomind.ingestion.github_fetcher import GitHubFetcher
from repomind.ingestion.service import IngestionService


def get_ingestion_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> IngestionService:
    fetcher = GitHubFetcher(settings, DefaultFileFilter(settings.max_file_size_bytes))
    chunker = CodeChunker(settings.chunk_max_lines, settings.chunk_max_chars)
    return IngestionService(fetcher, chunker)
