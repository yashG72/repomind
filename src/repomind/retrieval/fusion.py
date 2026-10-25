"""Reciprocal Rank Fusion (RRF): merge several ranked result lists into one.

Semantic search scores are cosine similarities (about -1 to 1); keyword search scores are
BM25 values (0 to unbounded). The numbers are on different scales, so comparing or adding
them directly would let one method drown out the other. RRF ignores the raw scores and uses
only each chunk's *position* in each list:

    rrf_score(chunk) = sum over lists containing it of  1 / (k + rank)

where rank starts at 1. A chunk ranked well by both methods beats a chunk ranked well by just
one. The constant k (60 in the original paper) smooths the curve, so the difference between
rank 1 and rank 2 is not overwhelming.

`reciprocal_rank_fusion` is a pure function: no I/O, no model, same input -> same output.
"""

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field

from repomind.core.models import CodeChunk, ScoredChunk

DEFAULT_RRF_K = 60


class FusedCandidate(BaseModel):
    """A chunk after fusion, with what each retriever said about it (for debugging)."""

    chunk: CodeChunk
    rrf_score: float
    ranks: dict[str, int] = Field(default_factory=dict)  # retriever name -> 1-based rank
    scores: dict[str, float] = Field(default_factory=dict)  # retriever name -> its raw score


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[ScoredChunk]],
    k: int = DEFAULT_RRF_K,
    limit: int | None = None,
) -> list[FusedCandidate]:
    """Fuse ranked lists (each best-first) keyed by retriever name, e.g. "semantic".

    - Chunks are identified by `chunk_id`, which is derived from (repository, commit, path,
      line range). The same lines in another repository or commit are a different chunk, so
      fusion never merges across repositories or commits.
    - If a list contains the same chunk more than once, only its best (first) rank counts.
    - Ties are broken by the best rank in any list, then by chunk id, so the order is
      deterministic.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    fused: dict[str, FusedCandidate] = {}
    for name, items in ranked_lists.items():
        rank = 0
        seen: set[str] = set()
        for item in items:
            chunk_id = item.chunk.chunk_id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            rank += 1
            candidate = fused.get(chunk_id)
            if candidate is None:
                candidate = fused[chunk_id] = FusedCandidate(chunk=item.chunk, rrf_score=0.0)
            candidate.rrf_score += 1.0 / (k + rank)
            candidate.ranks[name] = rank
            candidate.scores[name] = item.score

    ordered = sorted(
        fused.values(),
        # Rounding makes 1/61 + 1/62 and 1/62 + 1/61 compare equal despite float rounding.
        key=lambda c: (-round(c.rrf_score, 12), min(c.ranks.values()), c.chunk.chunk_id),
    )
    return ordered if limit is None else ordered[:limit]
