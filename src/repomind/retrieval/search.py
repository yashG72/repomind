"""Search: find the chunks of one repository that best answer a question.

Two modes:
- "semantic" (Phase 3): embed the question and return the nearest chunks by cosine
  similarity. Unchanged, for backward compatibility and evaluation.
- "hybrid" (default): run semantic search and keyword (BM25) search over the same
  repository and commit, merge the two ranked lists with Reciprocal Rank Fusion, then let a
  cross-encoder reranker reorder the strongest candidates.

    question ─► validate ─► semantic search ─┐
                       └──► keyword search ──┴─► RRF fusion ─► top N ─► reranker ─► top_k
"""

import logging
import time
from enum import StrEnum

from pydantic import BaseModel, Field

from repomind.config import Settings
from repomind.core.interfaces import Embedder, Reranker, VectorStore
from repomind.core.models import CodeChunk, RepoRef, ScoredChunk
from repomind.ingestion.github_url import parse_github_repo_url
from repomind.reranking.errors import RerankerError
from repomind.retrieval.errors import InvalidQuery, RepositoryNotIndexed
from repomind.retrieval.fusion import DEFAULT_RRF_K, FusedCandidate, reciprocal_rank_fusion

logger = logging.getLogger(__name__)

KEYWORD_UNAVAILABLE_WARNING = (
    "Keyword search is unavailable: this Qdrant collection was created before hybrid search "
    "existed, so these results come from semantic search only. To enable keyword search, set "
    "REPOMIND_QDRANT_COLLECTION to a new name and index the repository again."
)


class SearchMode(StrEnum):
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class SearchConfig(BaseModel, frozen=True):
    """Retrieval knobs. Defaults match the Settings defaults."""

    default_mode: SearchMode = SearchMode.HYBRID
    semantic_candidates: int = Field(default=50, ge=1)
    keyword_candidates: int = Field(default=50, ge=1)
    rrf_k: int = Field(default=DEFAULT_RRF_K, ge=1)
    rerank_enabled: bool = False  # the default for requests that do not choose
    rerank_max_candidates: int = Field(default=20, ge=1)

    @classmethod
    def from_settings(cls, settings: Settings) -> "SearchConfig":
        return cls(
            default_mode=SearchMode(settings.search_mode),
            semantic_candidates=settings.search_semantic_candidates,
            keyword_candidates=settings.search_keyword_candidates,
            rrf_k=settings.search_rrf_k,
            rerank_enabled=settings.rerank_enabled,
            rerank_max_candidates=settings.rerank_max_candidates,
        )


class HitScores(BaseModel):
    """What each stage said about a result. A stage that did not run, or did not find this
    chunk, is null. Useful for debugging and evaluation, not for comparing across queries."""

    semantic: float | None = None  # cosine similarity
    semantic_rank: int | None = None  # 1-based position in the semantic list
    keyword: float | None = None  # BM25 score
    keyword_rank: int | None = None  # 1-based position in the keyword list
    rrf: float | None = None  # Reciprocal Rank Fusion score
    rerank: float | None = None  # cross-encoder relevance (a logit: higher is more relevant)


class SearchHit(BaseModel):
    # The score that produced this ranking: cosine similarity in semantic mode; in hybrid
    # mode the reranker score if the hit was reranked, otherwise its RRF score.
    score: float
    path: str
    start_line: int
    end_line: int
    symbol: str | None
    language: str | None
    commit_sha: str
    url: str  # GitHub link to the exact lines at the indexed commit
    text: str
    scores: HitScores = Field(default_factory=HitScores)


class SearchResponse(BaseModel):
    repository: str
    query: str
    embedding_model: str
    results: list[SearchHit]
    duration_ms: float
    mode: SearchMode = SearchMode.SEMANTIC
    reranker_model: str | None = None  # set when results were reranked
    warnings: list[str] = Field(default_factory=list)


class SearchService:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker | None = None,
        config: SearchConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._reranker = reranker
        self._config = config or SearchConfig()

    def search(
        self,
        repo_url: str,
        query: str,
        top_k: int,
        commit_sha: str | None = None,
        mode: SearchMode | None = None,
        rerank: bool | None = None,
    ) -> SearchResponse:
        """Raises InvalidRepositoryURL, InvalidQuery, RepositoryNotIndexed, and EmbeddingError,
        RerankerError or VectorStoreError when a dependency fails."""
        started = time.perf_counter()
        repo = parse_github_repo_url(repo_url)
        query = query.strip()
        if not query:
            raise InvalidQuery("The query must not be empty")
        mode = mode or self._config.default_mode
        use_reranker = self._should_rerank(mode, rerank)
        if self._store.count(repo, commit_sha=commit_sha) == 0:
            raise RepositoryNotIndexed(
                f"{repo.full_name} has not been indexed"
                + (f" at commit {commit_sha}" if commit_sha else "")
            )

        timings: dict[str, float] = {}
        if mode is SearchMode.SEMANTIC:
            semantic = self._semantic(repo, query, top_k, commit_sha, timings)
            hits = [
                _hit(
                    item.chunk,
                    round(item.score, 4),
                    HitScores(semantic=item.score, semantic_rank=i),
                )
                for i, item in enumerate(semantic, start=1)
            ]
            chunks = [item.chunk for item in semantic]
            return self._respond(repo, chunks, query, mode, hits, None, [], started, timings, {})

        warnings: list[str] = []
        semantic = self._semantic(
            repo, query, max(top_k, self._config.semantic_candidates), commit_sha, timings
        )
        keyword = self._keyword(
            repo, query, max(top_k, self._config.keyword_candidates), commit_sha, timings, warnings
        )
        fused = reciprocal_rank_fusion(
            {
                "semantic": _in_scope(semantic, repo, commit_sha),
                "keyword": _in_scope(keyword, repo, commit_sha),
            },
            k=self._config.rrf_k,
        )
        reranked: dict[str, float] = {}
        if use_reranker and fused:
            fused, reranked = self._rerank(query, fused, timings)
        hits = [_fused_hit(c, reranked.get(c.chunk.chunk_id)) for c in fused[:top_k]]
        counts = {"semantic": len(semantic), "keyword": len(keyword), "fused": len(fused)}
        return self._respond(
            repo,
            [c.chunk for c in fused],
            query,
            mode,
            hits,
            self._reranker.model_name if reranked and self._reranker else None,
            warnings,
            started,
            timings,
            counts,
        )

    # --- Stages -----------------------------------------------------------------------------

    def _should_rerank(self, mode: SearchMode, requested: bool | None) -> bool:
        if mode is SearchMode.SEMANTIC:
            if requested:
                raise InvalidQuery("Reranking is only available in hybrid mode")
            return False
        if requested is None:
            return self._config.rerank_enabled and self._reranker is not None
        if requested and self._reranker is None:
            raise InvalidQuery("Reranking is not available on this server")
        return requested

    def _semantic(
        self,
        repo: RepoRef,
        query: str,
        limit: int,
        commit_sha: str | None,
        timings: dict[str, float],
    ) -> list[ScoredChunk]:
        start = time.perf_counter()
        vector = self._embedder.embed_query(query)
        results = self._store.search(
            repo, vector, limit, self._embedder.model_name, commit_sha=commit_sha
        )
        timings["semantic_ms"] = _ms_since(start)
        return results

    def _keyword(
        self,
        repo: RepoRef,
        query: str,
        limit: int,
        commit_sha: str | None,
        timings: dict[str, float],
        warnings: list[str],
    ) -> list[ScoredChunk]:
        if not self._store.supports_keyword_search():
            warnings.append(KEYWORD_UNAVAILABLE_WARNING)
            return []
        start = time.perf_counter()
        results = self._store.keyword_search(repo, query, limit, commit_sha=commit_sha)
        timings["keyword_ms"] = _ms_since(start)
        return results

    def _rerank(
        self, query: str, fused: list[FusedCandidate], timings: dict[str, float]
    ) -> tuple[list[FusedCandidate], dict[str, float]]:
        """Rerank the top candidates; the rest keep their fused order after them."""
        assert self._reranker is not None  # noqa: S101 (guaranteed by _should_rerank)
        pool, rest = (
            fused[: self._config.rerank_max_candidates],
            fused[self._config.rerank_max_candidates :],
        )
        start = time.perf_counter()
        ordered = self._reranker.rerank(
            query, [ScoredChunk(chunk=c.chunk, score=c.rrf_score) for c in pool]
        )
        timings["rerank_ms"] = _ms_since(start)
        by_id = {c.chunk.chunk_id: c for c in pool}
        if len(ordered) != len(pool) or {o.chunk.chunk_id for o in ordered} != set(by_id):
            raise RerankerError("The reranking model returned a different set of results")
        scores = {o.chunk.chunk_id: o.score for o in ordered}
        return [by_id[o.chunk.chunk_id] for o in ordered] + rest, scores

    def _respond(
        self,
        repo: RepoRef,
        chunks: list[CodeChunk],
        query: str,
        mode: SearchMode,
        hits: list[SearchHit],
        reranker_model: str | None,
        warnings: list[str],
        started: float,
        timings: dict[str, float],
        counts: dict[str, int],
    ) -> SearchResponse:
        duration = _ms_since(started)
        # The query is user text and may be sensitive, so only its length is logged.
        logger.info(
            "search completed",
            extra={
                "repo": repo.full_name,
                "mode": str(mode),
                "reranked": reranker_model is not None,
                "query_chars": len(query),
                "results": len(hits),
                "duration_ms": duration,
                **{f"{name}_candidates": n for name, n in counts.items()},
                **timings,
            },
        )
        return SearchResponse(
            # GitHub's own spelling of owner/name, as stored at indexing time.
            repository=chunks[0].repo.full_name if chunks else repo.full_name,
            query=query,
            embedding_model=self._embedder.model_name,
            results=hits,
            duration_ms=duration,
            mode=mode,
            reranker_model=reranker_model,
            warnings=warnings,
        )


def _in_scope(items: list[ScoredChunk], repo: RepoRef, commit_sha: str | None) -> list[ScoredChunk]:
    """Defense in depth: the store already filters by repository and commit, but never let a
    chunk from another repository or commit through even if a store misbehaves."""
    key = repo.full_name.lower()
    kept = [
        item
        for item in items
        if item.chunk.repo.full_name.lower() == key
        and (commit_sha is None or item.chunk.commit_sha == commit_sha)
    ]
    if len(kept) != len(items):
        logger.warning(
            "dropped out-of-scope search results", extra={"count": len(items) - len(kept)}
        )
    return kept


def _hit(chunk: CodeChunk, score: float, scores: HitScores) -> SearchHit:
    return SearchHit(
        score=score,
        path=chunk.path,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        symbol=chunk.symbol,
        language=chunk.language,
        commit_sha=chunk.commit_sha,
        url=chunk.github_url,
        text=chunk.text,
        scores=scores,
    )


def _fused_hit(candidate: FusedCandidate, rerank_score: float | None) -> SearchHit:
    scores = HitScores(
        semantic=candidate.scores.get("semantic"),
        semantic_rank=candidate.ranks.get("semantic"),
        keyword=candidate.scores.get("keyword"),
        keyword_rank=candidate.ranks.get("keyword"),
        rrf=round(candidate.rrf_score, 6),
        rerank=rerank_score,
    )
    final = rerank_score if rerank_score is not None else candidate.rrf_score
    return _hit(candidate.chunk, round(final, 6), scores)


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
