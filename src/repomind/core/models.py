"""Data types shared between pipeline components.

These are deliberately small. Every chunk keeps its file path and line range so answers
can always cite exactly where in the repository a piece of code came from.
"""

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


class CodeChunk(BaseModel):
    """A contiguous slice of one file, with 1-based inclusive line numbers."""

    chunk_id: str
    repo: RepoRef
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    language: str | None = None
    text: str

    @model_validator(mode="after")
    def _check_line_range(self) -> "CodeChunk":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self


class ScoredChunk(BaseModel):
    """A chunk returned by retrieval, with the score that ranked it."""

    chunk: CodeChunk
    score: float


class Citation(BaseModel):
    """A file reference attached to an answer."""

    path: str
    start_line: int
    end_line: int
    url: str
