"""Errors raised while ingesting a repository.

Each error carries a message that is safe to show to API users: it never includes tokens,
response bodies or file contents.
"""


class IngestionError(Exception):
    """Base class for all ingestion failures."""


class RepositoryNotFound(IngestionError):
    """The repository does not exist, is private, or is not visible to RepoMind."""


class RepositoryTooLarge(IngestionError):
    """The repository exceeds a configured size limit."""


class RateLimited(IngestionError):
    """GitHub refused the request because a rate limit was reached."""

    def __init__(self, message: str, retry_after_seconds: int | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class GitHubUnavailable(IngestionError):
    """GitHub could not be reached, timed out, or returned a server error."""


class MalformedResponse(IngestionError):
    """GitHub returned something RepoMind could not safely interpret."""


class UnsafeRedirect(IngestionError):
    """GitHub redirected to a destination outside the allowlist."""
