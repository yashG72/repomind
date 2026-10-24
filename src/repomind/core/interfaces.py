"""Interfaces (typing.Protocol) for each pipeline responsibility.

A Protocol describes what a component must do without tying the rest of the code to one
implementation. For example, any class with an `embed_texts` method of the right shape
counts as an Embedder, so tests can use a tiny fake and production can use
Sentence Transformers. Implementations arrive in later phases.
"""

from collections.abc import Sequence
from typing import Protocol

from repomind.core.models import CodeChunk, RepoRef, ScoredChunk, SourceFile


class RepositoryFetcher(Protocol):
    """Downloads a repository's files without executing anything from it."""

    def fetch(self, repo: RepoRef) -> list[SourceFile]: ...


class FileFilter(Protocol):
    """Decides which files are worth indexing (skips binaries, secrets, vendored code)."""

    def is_indexable(self, path: str, size_bytes: int) -> bool: ...


class Chunker(Protocol):
    """Splits a file into line-numbered chunks."""

    def chunk(self, repo: RepoRef, file: SourceFile) -> list[CodeChunk]: ...


class Embedder(Protocol):
    """Turns text into fixed-length vectors."""

    @property
    def dimension(self) -> int: ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    """Stores chunk vectors and finds the nearest ones to a query vector."""

    def upsert(self, chunks: Sequence[CodeChunk], vectors: Sequence[Sequence[float]]) -> None: ...

    def search(self, repo: RepoRef, vector: Sequence[float], limit: int) -> list[ScoredChunk]: ...


class Reranker(Protocol):
    """Reorders candidate chunks by relevance to the question."""

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]: ...


class LLMClient(Protocol):
    """Generates text from a prompt. Swappable between providers."""

    def generate(self, system_prompt: str, user_prompt: str) -> str: ...
