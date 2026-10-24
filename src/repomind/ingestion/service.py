"""Orchestrates one ingestion: validate URL -> fetch -> redact -> chunk -> summarize.

The service holds no repository data between calls; every result is returned to the caller.
Storing chunks (embeddings and Qdrant) is added in Phase 3.
"""

import logging
import time
from collections import Counter

from pydantic import BaseModel

from repomind.core.interfaces import Chunker, RepositoryFetcher
from repomind.core.models import CodeChunk, FetchResult, SkippedFile
from repomind.ingestion.github_url import parse_github_repo_url
from repomind.ingestion.secrets import redact_secrets

logger = logging.getLogger(__name__)

MAX_SKIPPED_IN_SUMMARY = 100


class IngestionSummary(BaseModel):
    repository: str
    url: str
    commit_sha: str
    default_branch: str
    files_discovered: int
    files_processed: int
    files_skipped: int
    chunks_created: int
    skipped_by_reason: dict[str, int]
    languages: dict[str, int]  # files processed per language
    secrets_redacted: int
    bytes_downloaded: int
    duration_ms: float
    # A sample of skipped files, so a user can see why something is missing.
    skipped_files: list[SkippedFile]
    skipped_files_truncated: bool


class IngestionResult(BaseModel):
    summary: IngestionSummary
    chunks: list[CodeChunk]


class IngestionService:
    def __init__(self, fetcher: RepositoryFetcher, chunker: Chunker) -> None:
        self._fetcher = fetcher
        self._chunker = chunker

    def ingest(self, repo_url: str) -> IngestionResult:
        """Raises InvalidRepositoryURL or an IngestionError subclass on failure."""
        started = time.perf_counter()
        repo = parse_github_repo_url(repo_url)
        fetched = self._fetcher.fetch(repo)

        chunks: list[CodeChunk] = []
        redactions = 0
        for source in fetched.files:
            content, count = redact_secrets(source.content)
            if count:
                redactions += count
                source = source.model_copy(update={"content": content})
            chunks.extend(self._chunker.chunk(repo, fetched.commit_sha, source))

        summary = _summarize(fetched, chunks, redactions, started)
        logger.info(
            "ingestion completed",
            extra={
                "repo": summary.repository,
                "commit_sha": summary.commit_sha,
                "files_processed": summary.files_processed,
                "files_skipped": summary.files_skipped,
                "chunks_created": summary.chunks_created,
                "secrets_redacted": redactions,
                "duration_ms": summary.duration_ms,
            },
        )
        return IngestionResult(summary=summary, chunks=chunks)


def _summarize(
    fetched: FetchResult, chunks: list[CodeChunk], redactions: int, started: float
) -> IngestionSummary:
    reasons = Counter(skip.reason.value for skip in fetched.skipped)
    languages = Counter(f.language or "unknown" for f in fetched.files)
    return IngestionSummary(
        repository=fetched.repo.full_name,
        url=fetched.repo.html_url,
        commit_sha=fetched.commit_sha,
        default_branch=fetched.default_branch,
        files_discovered=fetched.files_discovered,
        files_processed=len(fetched.files),
        files_skipped=len(fetched.skipped),
        chunks_created=len(chunks),
        skipped_by_reason=dict(sorted(reasons.items())),
        languages=dict(languages.most_common()),
        secrets_redacted=redactions,
        bytes_downloaded=fetched.bytes_downloaded,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
        skipped_files=fetched.skipped[:MAX_SKIPPED_IN_SUMMARY],
        skipped_files_truncated=len(fetched.skipped) > MAX_SKIPPED_IN_SUMMARY,
    )
