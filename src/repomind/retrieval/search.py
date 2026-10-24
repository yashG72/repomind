"""Semantic search: embed a question and return the most similar chunks of one repository."""

import logging
import time

from pydantic import BaseModel

from repomind.core.interfaces import Embedder, VectorStore
from repomind.ingestion.github_url import parse_github_repo_url
from repomind.retrieval.errors import InvalidQuery, RepositoryNotIndexed

logger = logging.getLogger(__name__)


class SearchHit(BaseModel):
    score: float  # cosine similarity, higher is more similar (at most 1.0)
    path: str
    start_line: int
    end_line: int
    symbol: str | None
    language: str | None
    commit_sha: str
    url: str  # GitHub link to the exact lines at the indexed commit
    text: str


class SearchResponse(BaseModel):
    repository: str
    query: str
    embedding_model: str
    results: list[SearchHit]
    duration_ms: float


class SearchService:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def search(
        self, repo_url: str, query: str, top_k: int, commit_sha: str | None = None
    ) -> SearchResponse:
        """Raises InvalidRepositoryURL, InvalidQuery, or RepositoryNotIndexed."""
        started = time.perf_counter()
        repo = parse_github_repo_url(repo_url)
        query = query.strip()
        if not query:
            raise InvalidQuery("The query must not be empty")
        if self._store.count(repo, commit_sha=commit_sha) == 0:
            raise RepositoryNotIndexed(
                f"{repo.full_name} has not been indexed"
                + (f" at commit {commit_sha}" if commit_sha else "")
            )

        vector = self._embedder.embed_query(query)
        scored = self._store.search(
            repo, vector, top_k, self._embedder.model_name, commit_sha=commit_sha
        )
        hits = [
            SearchHit(
                score=round(item.score, 4),
                path=item.chunk.path,
                start_line=item.chunk.start_line,
                end_line=item.chunk.end_line,
                symbol=item.chunk.symbol,
                language=item.chunk.language,
                commit_sha=item.chunk.commit_sha,
                url=item.chunk.github_url,
                text=item.chunk.text,
            )
            for item in scored
        ]
        duration = round((time.perf_counter() - started) * 1000, 2)
        # The query is user text and may be sensitive, so only its length is logged.
        logger.info(
            "search completed",
            extra={
                "repo": repo.full_name,
                "query_chars": len(query),
                "results": len(hits),
                "duration_ms": duration,
            },
        )
        return SearchResponse(
            repository=scored[0].chunk.repo.full_name if scored else repo.full_name,
            query=query,
            embedding_model=self._embedder.model_name,
            results=hits,
            duration_ms=duration,
        )
