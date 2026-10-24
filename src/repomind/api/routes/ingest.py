"""POST /ingest: fetch, filter and chunk a public GitHub repository, and report what happened.

In Phase 2 the chunks are not stored yet; the response is a summary. Phase 3 will store
them in Qdrant.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from repomind.api.dependencies import get_ingestion_service
from repomind.ingestion.errors import (
    GitHubUnavailable,
    IngestionError,
    RateLimited,
    RepositoryNotFound,
    RepositoryTooLarge,
)
from repomind.ingestion.github_url import InvalidRepositoryURL
from repomind.ingestion.service import IngestionService, IngestionSummary

router = APIRouter(tags=["ingestion"])


class IngestRequest(BaseModel):
    repo_url: str = Field(max_length=300, examples=["https://github.com/pallets/itsdangerous"])


@router.post("/ingest", response_model=IngestionSummary)
def ingest(
    request: IngestRequest,
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
) -> IngestionSummary:
    # A plain `def` endpoint: FastAPI runs it in a worker thread, so the blocking download
    # does not stall other requests.
    try:
        return service.ingest(request.repo_url).summary
    except InvalidRepositoryURL as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except RepositoryNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except RepositoryTooLarge as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(exc)) from exc
    except RateLimited as exc:
        headers = (
            {"Retry-After": str(exc.retry_after_seconds)}
            if exc.retry_after_seconds is not None
            else None
        )
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(exc), headers) from exc
    except GitHubUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except IngestionError as exc:  # malformed response, unsafe redirect
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
