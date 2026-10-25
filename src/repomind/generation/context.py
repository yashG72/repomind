"""Evidence context: turns search hits into the numbered sources an answer may cite.

Each source keeps its retrieval metadata (path, line range, commit, GitHub URL) and gets a
stable id, S1, S2, ... in rank order. The context is bounded by a number of chunks and a total
number of characters of code. A chunk is never cut: cutting would leave a line range that no
longer matches the text, so a citation could point at code the model never saw. A chunk that
does not fit is left out whole, and a smaller, lower-ranked chunk may take its place.

Nothing here knows about any LLM provider.
"""

from pydantic import BaseModel, Field

from repomind.retrieval.search import SearchHit


class Evidence(BaseModel, frozen=True):
    """One retrieved chunk, as shown to the model."""

    id: str  # "S1", "S2", ...: the only references the model may cite
    repository: str
    path: str
    start_line: int
    end_line: int
    symbol: str | None
    language: str | None
    commit_sha: str
    url: str
    score: float
    text: str


class EvidenceContext(BaseModel):
    evidence: list[Evidence]
    duplicates_dropped: int = Field(default=0, ge=0)
    dropped_over_budget: int = Field(default=0, ge=0)

    @property
    def chars(self) -> int:
        return sum(len(e.text) for e in self.evidence)

    def by_id(self) -> dict[str, Evidence]:
        return {e.id: e for e in self.evidence}


class ContextBuilder:
    def __init__(self, max_chunks: int, max_chars: int) -> None:
        if max_chunks < 1 or max_chars < 1:
            raise ValueError("context limits must be positive")
        self._max_chunks = max_chunks
        self._max_chars = max_chars

    def build(self, repository: str, hits: list[SearchHit]) -> EvidenceContext:
        """Select hits in rank order until the chunk or character budget is used up."""
        selected: list[SearchHit] = []
        duplicates = over_budget = 0
        remaining = self._max_chars
        for hit in hits:
            if len(selected) == self._max_chunks:
                break
            if any(_redundant(hit, kept) for kept in selected):
                duplicates += 1
                continue
            if len(hit.text) > remaining:
                over_budget += 1
                continue
            selected.append(hit)
            remaining -= len(hit.text)
        evidence = [
            Evidence(
                id=f"S{i}",
                repository=repository,
                path=hit.path,
                start_line=hit.start_line,
                end_line=hit.end_line,
                symbol=hit.symbol,
                language=hit.language,
                commit_sha=hit.commit_sha,
                url=hit.url,
                score=hit.score,
                text=hit.text,
            )
            for i, hit in enumerate(selected, start=1)
        ]
        return EvidenceContext(
            evidence=evidence, duplicates_dropped=duplicates, dropped_over_budget=over_budget
        )


def _redundant(hit: SearchHit, kept: SearchHit) -> bool:
    """True if `hit` adds nothing over `kept`: the same lines (or a sub-range) of the same file
    at the same commit, or byte-identical code such as a copied file."""
    if hit.text.strip() == kept.text.strip():
        return True
    return (
        hit.path == kept.path
        and hit.commit_sha == kept.commit_sha
        and kept.start_line <= hit.start_line
        and hit.end_line <= kept.end_line
    )
