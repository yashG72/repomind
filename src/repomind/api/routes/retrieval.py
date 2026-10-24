"""POST /index stores a repository's chunks as vectors; POST /search queries them."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from repomind.api.dependencies import get_indexing_service, get_search_service
from repomind.config import Settings, get_settings
from repomind.retrieval.indexing import IndexingService, IndexingSummary
from repomind.retrieval.search import SearchResponse, SearchService

router = APIRouter(tags=["retrieval"])


class IndexRequest(BaseModel):
    repo_url: str = Field(max_length=300, examples=["https://github.com/yashG72/repomind"])


class SearchRequest(BaseModel):
    repo_url: str = Field(max_length=300, examples=["https://github.com/yashG72/repomind"])
    query: str = Field(min_length=1, max_length=20_000, examples=["Where are URLs validated?"])
    top_k: int | None = Field(default=None, ge=1, le=50)
    # Optional: search one specific indexed commit instead of the current one.
    commit_sha: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")


@router.post("/index", response_model=IndexingSummary)
def index(
    request: IndexRequest,
    service: Annotated[IndexingService, Depends(get_indexing_service)],
) -> IndexingSummary:
    return service.index(request.repo_url)


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    service: Annotated[SearchService, Depends(get_search_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> SearchResponse:
    if len(request.query) > settings.search_max_query_chars:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"query is longer than {settings.search_max_query_chars} characters",
        )
    top_k = request.top_k or settings.search_default_top_k
    return service.search(request.repo_url, request.query, top_k, request.commit_sha)
