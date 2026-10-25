"""Maps domain errors to HTTP responses in one place, so routes only call services.

Error messages are written to be safe to show users: they never contain tokens, source
code, or upstream response bodies.
"""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from repomind.embeddings.errors import EmbeddingError
from repomind.generation.errors import (
    LLMAuthenticationError,
    LLMError,
    LLMNotConfigured,
    LLMRateLimited,
    LLMRefused,
    LLMTimeout,
    LLMUnavailable,
    MalformedLLMOutput,
)
from repomind.ingestion.errors import (
    GitHubUnavailable,
    IngestionError,
    RateLimited,
    RepositoryNotFound,
    RepositoryTooLarge,
)
from repomind.ingestion.github_url import InvalidRepositoryURL
from repomind.reranking.errors import RerankerError
from repomind.retrieval.errors import InvalidQuery, RepositoryNotIndexed
from repomind.vectorstore.errors import CollectionMismatch, VectorStoreError

logger = logging.getLogger(__name__)

# Checked in order; the first matching type wins (so subclasses come before base classes).
_STATUS_BY_ERROR: list[tuple[type[Exception], int]] = [
    (InvalidRepositoryURL, status.HTTP_422_UNPROCESSABLE_CONTENT),
    (InvalidQuery, status.HTTP_422_UNPROCESSABLE_CONTENT),
    (RepositoryNotFound, status.HTTP_404_NOT_FOUND),
    (RepositoryNotIndexed, status.HTTP_404_NOT_FOUND),
    (RepositoryTooLarge, status.HTTP_413_CONTENT_TOO_LARGE),
    (RateLimited, status.HTTP_429_TOO_MANY_REQUESTS),
    (GitHubUnavailable, status.HTTP_503_SERVICE_UNAVAILABLE),
    (IngestionError, status.HTTP_502_BAD_GATEWAY),  # malformed response, unsafe redirect
    (EmbeddingError, status.HTTP_503_SERVICE_UNAVAILABLE),
    (RerankerError, status.HTTP_503_SERVICE_UNAVAILABLE),
    (CollectionMismatch, status.HTTP_500_INTERNAL_SERVER_ERROR),
    (VectorStoreError, status.HTTP_503_SERVICE_UNAVAILABLE),
    (LLMNotConfigured, status.HTTP_503_SERVICE_UNAVAILABLE),
    (LLMRateLimited, status.HTTP_429_TOO_MANY_REQUESTS),
    (LLMTimeout, status.HTTP_504_GATEWAY_TIMEOUT),
    (LLMUnavailable, status.HTTP_503_SERVICE_UNAVAILABLE),
    (LLMRefused, status.HTTP_422_UNPROCESSABLE_CONTENT),
    # Our credentials or model were rejected, or the model replied in the wrong format.
    (LLMAuthenticationError, status.HTTP_502_BAD_GATEWAY),
    (MalformedLLMOutput, status.HTTP_502_BAD_GATEWAY),
    (LLMError, status.HTTP_502_BAD_GATEWAY),
]


async def _handle(request: Request, exc: Exception) -> JSONResponse:
    code = next(code for error_type, code in _STATUS_BY_ERROR if isinstance(exc, error_type))
    headers = None
    if isinstance(exc, RateLimited) and exc.retry_after_seconds is not None:
        headers = {"Retry-After": str(exc.retry_after_seconds)}
    logger.warning(
        "request failed",
        extra={"path": request.url.path, "error_type": type(exc).__name__, "status_code": code},
    )
    return JSONResponse({"detail": str(exc)}, status_code=code, headers=headers)


def register_exception_handlers(app: FastAPI) -> None:
    for error_type in {error_type for error_type, _ in _STATUS_BY_ERROR}:
        app.add_exception_handler(error_type, _handle)
