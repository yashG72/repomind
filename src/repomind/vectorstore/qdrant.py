"""Qdrant implementation of the VectorStore protocol.

Two ways to run Qdrant, chosen by settings:
- Local mode (default): `qdrant_path` is a folder on disk. No server is needed, and data
  persists between restarts. Only one process can open the folder at a time.
- Server mode: `qdrant_url` points to a Qdrant server (Docker or Qdrant Cloud), with an
  optional API key.

Data layout: one collection holds the chunks of every repository. Each point is one chunk:
- id: a UUID derived from (repository, commit, path, line range), so indexing the same commit
  twice overwrites the same points instead of creating duplicates.
- vector: the chunk's embedding (cosine distance), stored as the collection's unnamed vector.
- "keywords" sparse vector: BM25 term weights from `repomind.retrieval.keywords`, for keyword
  search. Qdrant applies the IDF part itself (the sparse vector uses its IDF modifier).
- payload: repository, commit SHA, path, symbol, line range, language, the (already
  secret-redacted) chunk text, the embedding model name and the keyword encoding version.

Every query filters on `repo_key` (the lower-cased "owner/name"), so a search can never
return another repository's chunks.

Collections created before Phase 4 have no "keywords" sparse vector, and Qdrant cannot add
one to an existing collection. They keep working for semantic search;
`supports_keyword_search()` reports False for them, and keyword search needs a new
collection (set REPOMIND_QDRANT_COLLECTION to a new name and re-index).
"""

import logging
import threading
import uuid
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from qdrant_client import QdrantClient, models

from repomind.config import Settings
from repomind.core.models import CodeChunk, RepoRef, ScoredChunk, document_text
from repomind.retrieval.keywords import KEYWORD_ENCODING_VERSION, encode_document, encode_query
from repomind.vectorstore.errors import (
    CollectionMismatch,
    KeywordSearchUnavailable,
    VectorStoreError,
    VectorStoreUnavailable,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

_KEYWORD_INDEXES = ("repo_key", "commit_sha", "embedding_model", "path")
KEYWORD_VECTOR = "keywords"


def point_id(chunk_id: str) -> str:
    """Qdrant ids must be integers or UUIDs; chunk ids are 32 hex characters, i.e. a UUID."""
    return str(uuid.UUID(hex=chunk_id))


def repo_key(repo: RepoRef) -> str:
    # GitHub names are case-insensitive, so "Octo/Demo" and "octo/demo" are one repository.
    return repo.full_name.lower()


class _Payload(BaseModel):
    chunk_id: str
    repo_owner: str
    repo_name: str
    commit_sha: str
    path: str
    start_line: int
    end_line: int
    language: str | None = None
    symbol: str | None = None
    text: str


class QdrantVectorStore:
    def __init__(
        self,
        client_factory: Callable[[], QdrantClient],
        collection: str,
        payload_indexes: bool = True,
    ) -> None:
        self._client_factory = client_factory
        # Payload indexes speed up the repo filter on a Qdrant server; local mode ignores them.
        self._payload_indexes = payload_indexes
        self._client: QdrantClient | None = None
        self._lock = threading.Lock()
        self.collection = collection
        self._ready_dimension: int | None = None
        self._has_keywords: bool | None = None  # None = not checked yet

    @classmethod
    def from_settings(cls, settings: Settings) -> "QdrantVectorStore":
        def factory() -> QdrantClient:
            if settings.qdrant_url:
                api_key = settings.qdrant_api_key
                return QdrantClient(
                    url=settings.qdrant_url,
                    api_key=api_key.get_secret_value() if api_key else None,
                    timeout=settings.qdrant_timeout_seconds,
                )
            return QdrantClient(path=settings.qdrant_path)

        return cls(factory, settings.qdrant_collection, payload_indexes=bool(settings.qdrant_url))

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    # --- Protocol methods -------------------------------------------------------------------

    def ensure_collection(self, dimension: int) -> None:
        if self._ready_dimension == dimension:
            return

        def create_or_check(client: QdrantClient) -> None:
            if not client.collection_exists(self.collection):
                logger.info(
                    "creating collection",
                    extra={"collection": self.collection, "dimension": dimension},
                )
                client.create_collection(
                    self.collection,
                    vectors_config=models.VectorParams(
                        size=dimension, distance=models.Distance.COSINE
                    ),
                    sparse_vectors_config={
                        KEYWORD_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
                    },
                )
                if self._payload_indexes:
                    for field in _KEYWORD_INDEXES:
                        client.create_payload_index(
                            self.collection, field, field_schema=models.PayloadSchemaType.KEYWORD
                        )
                    client.create_payload_index(
                        self.collection,
                        "keyword_version",
                        field_schema=models.PayloadSchemaType.INTEGER,
                    )
                self._has_keywords = True
                return
            info = client.get_collection(self.collection)
            params = info.config.params.vectors
            if not isinstance(params, models.VectorParams):
                raise CollectionMismatch(
                    f"Collection '{self.collection}' uses named vectors; use another collection"
                )
            if params.size != dimension or params.distance != models.Distance.COSINE:
                raise CollectionMismatch(
                    f"Collection '{self.collection}' stores {params.size}-dimension "
                    f"{params.distance} vectors, but the embedding model produces {dimension}-"
                    "dimension vectors. Set REPOMIND_QDRANT_COLLECTION to a new name."
                )
            self._has_keywords = _has_keyword_vector(info)
            if not self._has_keywords:
                logger.warning(
                    "collection has no keyword index; keyword search is unavailable",
                    extra={"collection": self.collection},
                )

        self._call(create_or_check)
        self._ready_dimension = dimension

    def existing_ids(self, chunk_ids: Sequence[str], embedding_model: str) -> set[str]:
        """Which chunks are already stored with a vector from this embedding model (and, if
        the collection supports keyword search, with the current keyword encoding)."""
        if not chunk_ids:
            return set()
        keyword_version = KEYWORD_ENCODING_VERSION if self.supports_keyword_search() else None

        def retrieve(client: QdrantClient) -> set[str]:
            found: set[str] = set()
            for start in range(0, len(chunk_ids), 512):
                batch = [point_id(c) for c in chunk_ids[start : start + 512]]
                records = client.retrieve(
                    self.collection,
                    ids=batch,
                    with_payload=["chunk_id", "embedding_model", "keyword_version"],
                    with_vectors=False,
                )
                for record in records:
                    payload = record.payload or {}
                    if payload.get("embedding_model") == embedding_model and (
                        keyword_version is None or payload.get("keyword_version") == keyword_version
                    ):
                        found.add(str(payload.get("chunk_id")))
            return found

        return self._call(retrieve)

    def upsert(
        self,
        chunks: Sequence[CodeChunk],
        vectors: Sequence[Sequence[float]],
        embedding_model: str,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        keywords = self.supports_keyword_search()
        points = [
            models.PointStruct(
                id=point_id(chunk.chunk_id),
                vector=_vectors(chunk, vector) if keywords else list(vector),
                payload=_payload(chunk, embedding_model, keywords),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._call(lambda c: c.upsert(self.collection, points=points, wait=True))

    def delete_other_commits(self, repo: RepoRef, keep_commit_sha: str) -> int:
        """Delete this repository's points from any commit other than keep_commit_sha."""
        stale = models.Filter(
            must=[_match("repo_key", repo_key(repo))],
            must_not=[_match("commit_sha", keep_commit_sha)],
        )

        def delete(client: QdrantClient) -> int:
            count = client.count(self.collection, count_filter=stale, exact=True).count
            if count:
                client.delete(
                    self.collection,
                    points_selector=models.FilterSelector(filter=stale),
                    wait=True,
                )
            return count

        if not self._collection_exists():
            return 0
        return self._call(delete)

    def count(self, repo: RepoRef, commit_sha: str | None = None) -> int:
        if not self._collection_exists():
            return 0
        query = _repo_filter(repo, commit_sha=commit_sha)
        return self._call(lambda c: c.count(self.collection, count_filter=query, exact=True).count)

    def search(
        self,
        repo: RepoRef,
        vector: Sequence[float],
        limit: int,
        embedding_model: str,
        commit_sha: str | None = None,
    ) -> list[ScoredChunk]:
        if not self._collection_exists():
            return []
        query = _repo_filter(repo, commit_sha=commit_sha, embedding_model=embedding_model)
        response = self._call(
            lambda c: c.query_points(
                self.collection,
                query=list(vector),
                query_filter=query,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        )
        return self._scored(response.points, repo)

    def supports_keyword_search(self) -> bool:
        """True if the collection has the keyword sparse vector (created in Phase 4 or later).
        A collection that does not exist yet will be created with it, so that also counts."""
        if self._has_keywords is None:

            def check(client: QdrantClient) -> bool | None:
                if not client.collection_exists(self.collection):
                    return None  # not cached: decided when the collection is created
                return _has_keyword_vector(client.get_collection(self.collection))

            result = self._call(check)
            if result is None:
                return True
            self._has_keywords = result
        return self._has_keywords

    def keyword_search(
        self,
        repo: RepoRef,
        query: str,
        limit: int,
        commit_sha: str | None = None,
    ) -> list[ScoredChunk]:
        """BM25-ranked chunks of one repository (and optionally one commit) sharing terms with
        the query. Scores are BM25 scores: higher is better, not comparable to cosine."""
        if not self._collection_exists():
            return []
        if not self.supports_keyword_search():
            raise KeywordSearchUnavailable(
                f"Collection '{self.collection}' was created without a keyword index"
            )
        terms = encode_query(query)
        if not terms:
            return []  # nothing searchable in the query, e.g. only punctuation or stopwords
        query_filter = _repo_filter(
            repo, commit_sha=commit_sha, keyword_version=KEYWORD_ENCODING_VERSION
        )
        response = self._call(
            lambda c: c.query_points(
                self.collection,
                query=models.SparseVector(indices=terms.indices, values=terms.values),
                using=KEYWORD_VECTOR,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        )
        return self._scored(response.points, repo)

    # --- Helpers ----------------------------------------------------------------------------

    def _scored(self, points: Sequence[models.ScoredPoint], repo: RepoRef) -> list[ScoredChunk]:
        results: list[ScoredChunk] = []
        malformed = 0
        for point in points:
            chunk = _chunk_from_payload(point.payload)
            if chunk is None or repo_key(chunk.repo) != repo_key(repo):
                malformed += 1
                continue
            results.append(ScoredChunk(chunk=chunk, score=float(point.score)))
        if malformed:
            logger.warning(
                "skipped malformed search results",
                extra={"collection": self.collection, "count": malformed},
            )
        return results

    def _collection_exists(self) -> bool:
        return self._call(lambda c: c.collection_exists(self.collection))

    def _get_client(self) -> QdrantClient:
        with self._lock:
            if self._client is None:
                self._client = self._client_factory()
            return self._client

    def _call(self, operation: Callable[[QdrantClient], T]) -> T:
        """Run a Qdrant operation, turning any client/network failure into our own error."""
        try:
            return operation(self._get_client())
        except VectorStoreError:
            raise
        except Exception as exc:
            # Connection refused, timeouts, HTTP errors, a locked local folder, ...
            logger.error(
                "vector store operation failed",
                extra={"collection": self.collection, "error_type": type(exc).__name__},
            )
            raise VectorStoreUnavailable("The vector store is unavailable") from exc


def _match(key: str, value: str | int) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _repo_filter(
    repo: RepoRef,
    commit_sha: str | None = None,
    embedding_model: str | None = None,
    keyword_version: int | None = None,
) -> models.Filter:
    must: list[Any] = [_match("repo_key", repo_key(repo))]
    if commit_sha:
        must.append(_match("commit_sha", commit_sha))
    if embedding_model:
        must.append(_match("embedding_model", embedding_model))
    if keyword_version is not None:
        must.append(_match("keyword_version", keyword_version))
    return models.Filter(must=must)


def _has_keyword_vector(info: models.CollectionInfo) -> bool:
    return KEYWORD_VECTOR in (info.config.params.sparse_vectors or {})


def _vectors(chunk: CodeChunk, vector: Sequence[float]) -> dict[str, Any]:
    terms = encode_document(document_text(chunk))
    return {
        "": list(vector),
        KEYWORD_VECTOR: models.SparseVector(indices=terms.indices, values=terms.values),
    }


def _payload(chunk: CodeChunk, embedding_model: str, keywords: bool) -> dict[str, Any]:
    payload = {
        "chunk_id": chunk.chunk_id,
        "repo_key": repo_key(chunk.repo),
        "repo_owner": chunk.repo.owner,
        "repo_name": chunk.repo.name,
        "commit_sha": chunk.commit_sha,
        "path": chunk.path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "language": chunk.language,
        "symbol": chunk.symbol,
        "text": chunk.text,
        "embedding_model": embedding_model,
    }
    if keywords:
        payload["keyword_version"] = KEYWORD_ENCODING_VERSION
    return payload


def _chunk_from_payload(payload: dict[str, Any] | None) -> CodeChunk | None:
    try:
        data = _Payload.model_validate(payload or {})
        return CodeChunk(
            chunk_id=data.chunk_id,
            repo=RepoRef(owner=data.repo_owner, name=data.repo_name),
            commit_sha=data.commit_sha,
            path=data.path,
            start_line=data.start_line,
            end_line=data.end_line,
            language=data.language,
            symbol=data.symbol,
            text=data.text,
        )
    except (ValidationError, ValueError):
        return None
