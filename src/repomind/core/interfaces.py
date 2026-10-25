"""Interfaces (typing.Protocol) for each pipeline responsibility.

A Protocol describes what a component must do without tying the rest of the code to one
implementation. For example, any class with an `embed_texts` method of the right shape
counts as an Embedder, so tests can use a tiny fake and production can use
Sentence Transformers. Implementations arrive in later phases.
"""

from collections.abc import Sequence
from typing import Protocol

from repomind.core.models import (
    CodeChunk,
    FetchResult,
    RepoRef,
    ScoredChunk,
    SkipReason,
    SourceFile,
)


class RepositoryFetcher(Protocol):
    """Downloads a repository's files without executing anything from it.

    Returns the indexable files plus a record of every file it skipped and why.
    """

    def fetch(self, repo: RepoRef) -> FetchResult: ...


class FileFilter(Protocol):
    """Decides which files are worth indexing (skips binaries, secrets, vendored code).

    The check is split in two so a fetcher can reject a file by its path and size
    before reading its bytes at all.
    """

    def check_path(self, path: str, size_bytes: int) -> SkipReason | None: ...

    def check_content(self, path: str, data: bytes) -> SkipReason | None: ...

    def language_for(self, path: str) -> str | None: ...


class Chunker(Protocol):
    """Splits a file into line-numbered chunks."""

    def chunk(self, repo: RepoRef, commit_sha: str, file: SourceFile) -> list[CodeChunk]: ...


class Embedder(Protocol):
    """Turns text into fixed-length vectors.

    Documents (code chunks) and queries (questions) are embedded separately because some
    models expect different prefixes or instructions for each.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    """Stores chunk vectors and finds the nearest ones to a query vector. Also finds chunks
    by keyword (exact names and words), ranked with BM25.

    Every read and delete is scoped to one repository, so results never mix repositories.
    """

    def ensure_collection(self, dimension: int) -> None: ...

    def existing_ids(self, chunk_ids: Sequence[str], embedding_model: str) -> set[str]: ...

    def upsert(
        self,
        chunks: Sequence[CodeChunk],
        vectors: Sequence[Sequence[float]],
        embedding_model: str,
    ) -> None: ...

    def delete_other_commits(self, repo: RepoRef, keep_commit_sha: str) -> int: ...

    def count(self, repo: RepoRef, commit_sha: str | None = None) -> int: ...

    def search(
        self,
        repo: RepoRef,
        vector: Sequence[float],
        limit: int,
        embedding_model: str,
        commit_sha: str | None = None,
    ) -> list[ScoredChunk]: ...

    def supports_keyword_search(self) -> bool: ...

    def keyword_search(
        self,
        repo: RepoRef,
        query: str,
        limit: int,
        commit_sha: str | None = None,
    ) -> list[ScoredChunk]: ...


class Reranker(Protocol):
    """Reorders candidate chunks by relevance to the question.

    Returns the same chunks (no more, no fewer), best first, each with the reranker's own
    relevance score in place of the score it arrived with.
    """

    @property
    def model_name(self) -> str: ...

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]: ...


class LLMClient(Protocol):
    """Generates text from a prompt. Swappable between providers."""

    def generate(self, system_prompt: str, user_prompt: str) -> str: ...
