"""Indexing: ingest a repository, embed its chunks, and store them in the vector store.

Re-indexing is safe and cheap:
- Point ids are derived from (repository, commit, path, line range), so storing the same
  commit again overwrites points instead of duplicating them.
- Chunks already stored for this commit with the current embedding model are not
  re-embedded; only new chunks are.
- After the new commit's chunks are stored, chunks from older commits of the same
  repository are deleted, so a search sees one version of the code.
"""

import logging
import time
from collections.abc import Sequence

from pydantic import BaseModel

from repomind.core.interfaces import Embedder, VectorStore
from repomind.core.models import CodeChunk, document_text
from repomind.ingestion.service import IngestionService, IngestionSummary

logger = logging.getLogger(__name__)

__all__ = ["IndexingService", "IndexingSummary", "document_text"]


class IndexingSummary(BaseModel):
    ingestion: IngestionSummary
    collection: str
    embedding_model: str
    embedding_dimension: int
    chunks_indexed: int  # chunks now stored for this commit
    embeddings_computed: int
    embeddings_reused: int
    stale_chunks_deleted: int  # chunks from older commits that were removed
    duration_ms: float


class IndexingService:
    def __init__(
        self,
        ingestion: IngestionService,
        embedder: Embedder,
        store: VectorStore,
        collection_name: str,
        batch_size: int = 128,
    ) -> None:
        self._ingestion = ingestion
        self._embedder = embedder
        self._store = store
        self._collection = collection_name
        self._batch_size = batch_size

    def index(self, repo_url: str) -> IndexingSummary:
        started = time.perf_counter()
        result = self._ingestion.ingest(repo_url)
        summary = result.summary
        chunks = [c for c in result.chunks if c.text.strip()]

        dimension = self._embedder.dimension
        model = self._embedder.model_name
        self._store.ensure_collection(dimension)

        existing = self._store.existing_ids([c.chunk_id for c in chunks], model)
        missing = [c for c in chunks if c.chunk_id not in existing]
        for batch in _batches(missing, self._batch_size):
            vectors = self._embedder.embed_documents([document_text(c) for c in batch])
            self._store.upsert(batch, vectors, model)

        # Only after the new version is fully stored: remove other versions of this repo.
        # For an empty repository (no commit) this removes everything previously stored.
        stale = self._store.delete_other_commits(result.repo, summary.commit_sha)

        indexed = IndexingSummary(
            ingestion=summary,
            collection=self._collection,
            embedding_model=model,
            embedding_dimension=dimension,
            chunks_indexed=len(chunks),
            embeddings_computed=len(missing),
            embeddings_reused=len(chunks) - len(missing),
            stale_chunks_deleted=stale,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        logger.info(
            "repository indexed",
            extra={
                "repo": summary.repository,
                "commit_sha": summary.commit_sha,
                "chunks_indexed": indexed.chunks_indexed,
                "embeddings_computed": indexed.embeddings_computed,
                "embeddings_reused": indexed.embeddings_reused,
                "stale_chunks_deleted": stale,
                "duration_ms": indexed.duration_ms,
            },
        )
        return indexed


def _batches(items: Sequence[CodeChunk], size: int) -> list[Sequence[CodeChunk]]:
    return [items[i : i + size] for i in range(0, len(items), size)]
