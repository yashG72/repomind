"""Small, deterministic stand-ins for the embedding model and the vector store."""

import hashlib
import math
import re
from collections.abc import Sequence

from repomind.core.models import CodeChunk, RepoRef, ScoredChunk
from repomind.embeddings.errors import EmbeddingError

_WORD = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Hashes words into a fixed number of buckets: texts sharing words get similar vectors.

    Not a real language model, but deterministic and instant, which is what unit tests need.
    """

    def __init__(self, dimension: int = 64, model_name: str = "fake-model") -> None:
        self._dimension = dimension
        self._model_name = model_name
        self.documents_embedded = 0
        self.queries_embedded = 0
        self.fail = False

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.documents_embedded += len(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries_embedded += 1
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        if self.fail:
            raise EmbeddingError("The embedding model failed to embed the text")
        vector = [0.0] * self._dimension
        for word in _WORD.findall(text.lower()):
            bucket = int(hashlib.md5(word.encode(), usedforsecurity=False).hexdigest(), 16)
            vector[bucket % self._dimension] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        if not any(vector):
            vector[0] = 1.0
        return [x / norm for x in vector]


class InMemoryVectorStore:
    """A dictionary-backed VectorStore with the same scoping rules as the Qdrant store."""

    def __init__(self) -> None:
        self.points: dict[str, tuple[CodeChunk, list[float], str]] = {}
        self.dimension: int | None = None

    def ensure_collection(self, dimension: int) -> None:
        self.dimension = dimension

    def existing_ids(self, chunk_ids: Sequence[str], embedding_model: str) -> set[str]:
        return {c for c in chunk_ids if c in self.points and self.points[c][2] == embedding_model}

    def upsert(
        self,
        chunks: Sequence[CodeChunk],
        vectors: Sequence[Sequence[float]],
        embedding_model: str,
    ) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.points[chunk.chunk_id] = (chunk, list(vector), embedding_model)

    def delete_other_commits(self, repo: RepoRef, keep_commit_sha: str) -> int:
        stale = [
            key
            for key, (chunk, _, _) in self.points.items()
            if _same_repo(chunk.repo, repo) and chunk.commit_sha != keep_commit_sha
        ]
        for key in stale:
            del self.points[key]
        return len(stale)

    def count(self, repo: RepoRef, commit_sha: str | None = None) -> int:
        return sum(
            1
            for chunk, _, _ in self.points.values()
            if _same_repo(chunk.repo, repo) and commit_sha in (None, chunk.commit_sha)
        )

    def search(
        self,
        repo: RepoRef,
        vector: Sequence[float],
        limit: int,
        embedding_model: str,
        commit_sha: str | None = None,
    ) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(chunk=chunk, score=sum(a * b for a, b in zip(vector, stored, strict=True)))
            for chunk, stored, model in self.points.values()
            if _same_repo(chunk.repo, repo)
            and model == embedding_model
            and commit_sha in (None, chunk.commit_sha)
        ]
        return sorted(scored, key=lambda s: s.score, reverse=True)[:limit]


def _same_repo(a: RepoRef, b: RepoRef) -> bool:
    return a.full_name.lower() == b.full_name.lower()
