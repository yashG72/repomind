"""Data types shared between pipeline components.

These are deliberately small. Every chunk keeps its file path and line range so answers
can always cite exactly where in the repository a piece of code came from.
"""

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class RepoRef(BaseModel, frozen=True):
    """A validated reference to a public GitHub repository."""

    owner: str
    name: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def html_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}"


class SourceFile(BaseModel):
    """A text file fetched from a repository. Its content is untrusted input."""

    path: str
    content: str
    size_bytes: int = Field(ge=0)
    language: str | None = None


class SkipReason(StrEnum):
    """Why a file in the repository was not indexed."""

    UNSAFE_PATH = "unsafe_path"
    NOT_REGULAR_FILE = "not_regular_file"  # symlinks, hard links, devices
    EXCLUDED_DIRECTORY = "excluded_directory"  # node_modules, vendor, build output, ...
    UNSUPPORTED_TYPE = "unsupported_type"
    LOCK_FILE = "lock_file"
    SECRET_FILE = "secret_file"  # noqa: S105 (a label, not a password): .env, keys
    TOO_LARGE = "too_large"
    EMPTY = "empty"
    BINARY = "binary"
    NOT_UTF8 = "not_utf8"
    GENERATED = "generated"
    MINIFIED = "minified"
    FILE_LIMIT = "file_limit"  # max_files_per_repo reached


class SkippedFile(BaseModel):
    path: str
    reason: SkipReason


class FetchResult(BaseModel):
    """Everything a fetcher found in one repository snapshot."""

    repo: RepoRef
    commit_sha: str
    default_branch: str
    files: list[SourceFile]
    skipped: list[SkippedFile]
    files_discovered: int = Field(ge=0)
    bytes_downloaded: int = Field(ge=0)


class CodeChunk(BaseModel):
    """A contiguous slice of one file, with 1-based inclusive line numbers."""

    chunk_id: str
    repo: RepoRef
    commit_sha: str
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    language: str | None = None
    symbol: str | None = None  # e.g. "class Parser" or "def Parser.parse" when known
    text: str

    @model_validator(mode="after")
    def _check_line_range(self) -> "CodeChunk":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self

    @property
    def github_url(self) -> str:
        return (
            f"{self.repo.html_url}/blob/{self.commit_sha}/{self.path}"
            f"#L{self.start_line}-L{self.end_line}"
        )


def document_text(chunk: CodeChunk) -> str:
    """The text that represents a chunk for search (embedding, keywords, reranking). The path
    and symbol help match questions such as "where is the GitHub fetcher?" even when the code
    itself never says so."""
    header = chunk.path if chunk.symbol is None else f"{chunk.path}\n{chunk.symbol}"
    return f"{header}\n\n{chunk.text}"


class ScoredChunk(BaseModel):
    """A chunk returned by retrieval, with the score that ranked it."""

    chunk: CodeChunk
    score: float


class Citation(BaseModel):
    """A file reference attached to an answer. Every field comes from retrieval metadata,
    never from the model's own text."""

    id: str  # the evidence id the answer cites, e.g. "S1"
    repository: str
    path: str
    start_line: int
    end_line: int
    commit_sha: str
    url: str
    symbol: str | None = None
