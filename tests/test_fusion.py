"""Reciprocal Rank Fusion is a pure function, tested here without any store or model."""

import uuid

import pytest

from repomind.core.models import CodeChunk, RepoRef, ScoredChunk
from repomind.retrieval.fusion import reciprocal_rank_fusion

A = RepoRef(owner="octo", name="alpha")
B = RepoRef(owner="octo", name="beta")
SHA1, SHA2 = "1" * 40, "2" * 40


def chunk(path: str, repo: RepoRef = A, sha: str = SHA1) -> CodeChunk:
    return CodeChunk(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_URL, f"{repo.full_name}{sha}{path}").hex,
        repo=repo,
        commit_sha=sha,
        path=path,
        start_line=1,
        end_line=3,
        text="code",
    )


def scored(*items: tuple[CodeChunk, float]) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=c, score=s) for c, s in items]


X, Y, Z = chunk("x.py"), chunk("y.py"), chunk("z.py")


def test_chunk_in_both_lists_gets_both_contributions() -> None:
    fused = reciprocal_rank_fusion(
        {"semantic": scored((X, 0.9), (Y, 0.8)), "keyword": scored((Y, 12.0), (Z, 3.0))}, k=60
    )

    assert [c.chunk.path for c in fused] == ["y.py", "x.py", "z.py"]
    top = fused[0]
    assert top.rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert top.ranks == {"semantic": 2, "keyword": 1}
    assert top.scores == {"semantic": 0.8, "keyword": 12.0}  # raw scores kept for debugging
    assert fused[1].ranks == {"semantic": 1} and fused[2].ranks == {"keyword": 2}


def test_ranks_matter_not_raw_scores() -> None:
    # A huge BM25 score must not outweigh a cosine score just because its scale is larger.
    fused = reciprocal_rank_fusion(
        {"semantic": scored((X, 0.31)), "keyword": scored((Y, 1_000.0))}, k=60
    )

    assert fused[0].rrf_score == fused[1].rrf_score == pytest.approx(1 / 61)


@pytest.mark.parametrize("order", [("semantic", "keyword"), ("keyword", "semantic")])
def test_result_does_not_depend_on_list_order(order) -> None:
    lists = {"semantic": scored((X, 0.9), (Y, 0.5)), "keyword": scored((Y, 2.0), (X, 1.0))}

    fused = reciprocal_rank_fusion({name: lists[name] for name in order})

    # X and Y tie on RRF (1/61 + 1/62 each); ties break by best rank, then chunk id.
    assert fused[0].rrf_score == pytest.approx(fused[1].rrf_score)
    assert [c.chunk.chunk_id for c in fused] == sorted([X.chunk_id, Y.chunk_id])


def test_different_orders_change_the_winner() -> None:
    agree = reciprocal_rank_fusion(
        {"semantic": scored((X, 0.9), (Y, 0.8)), "keyword": scored((X, 5.0), (Y, 4.0))}
    )
    disagree = reciprocal_rank_fusion(
        {"semantic": scored((X, 0.9), (Y, 0.8), (Z, 0.1)), "keyword": scored((Z, 9.0), (Y, 1.0))}
    )

    assert [c.chunk.path for c in agree] == ["x.py", "y.py"]
    # Z: 1/63 + 1/61 = 0.03227 edges out Y: 1/62 + 1/62 = 0.03226; X (one list) is last.
    assert [c.chunk.path for c in disagree] == ["z.py", "y.py", "x.py"]


def test_duplicates_within_a_list_count_once_at_their_best_rank() -> None:
    fused = reciprocal_rank_fusion({"keyword": scored((X, 5.0), (X, 4.0), (Y, 3.0))})

    assert [c.chunk.path for c in fused] == ["x.py", "y.py"]
    assert fused[0].ranks == {"keyword": 1} and fused[0].scores == {"keyword": 5.0}
    assert fused[1].ranks == {"keyword": 2}  # the duplicate does not use up a rank


def test_empty_lists() -> None:
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"semantic": [], "keyword": []}) == []

    only = reciprocal_rank_fusion({"semantic": [], "keyword": scored((X, 1.0))})
    assert [c.chunk.path for c in only] == ["x.py"]


def test_tied_and_zero_scores_keep_rank_order() -> None:
    fused = reciprocal_rank_fusion({"keyword": scored((Z, 0.0), (X, 0.0), (Y, 0.0))})

    assert [c.chunk.path for c in fused] == ["z.py", "x.py", "y.py"]
    assert all(c.scores["keyword"] == 0.0 for c in fused)


def test_missing_scores_from_one_list_are_absent_not_zero() -> None:
    fused = reciprocal_rank_fusion({"semantic": scored((X, 0.5)), "keyword": scored((Y, 2.0))})

    by_path = {c.chunk.path: c for c in fused}
    assert "keyword" not in by_path["x.py"].scores and "keyword" not in by_path["x.py"].ranks
    assert "semantic" not in by_path["y.py"].scores


def test_same_lines_in_another_repository_or_commit_are_never_merged() -> None:
    here, other_repo, other_commit = chunk("x.py"), chunk("x.py", repo=B), chunk("x.py", sha=SHA2)

    fused = reciprocal_rank_fusion(
        {"semantic": scored((here, 0.9), (other_repo, 0.9)), "keyword": scored((other_commit, 1))}
    )

    assert len(fused) == 3
    assert {(c.chunk.repo, c.chunk.commit_sha) for c in fused} == {(A, SHA1), (B, SHA1), (A, SHA2)}
    assert all(len(c.ranks) == 1 for c in fused)


def test_k_is_configurable_and_validated() -> None:
    lists = {"semantic": scored((X, 1.0)), "keyword": scored((X, 1.0))}

    assert reciprocal_rank_fusion(lists, k=1)[0].rrf_score == pytest.approx(1.0)
    assert reciprocal_rank_fusion(lists, k=60)[0].rrf_score == pytest.approx(2 / 61)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(lists, k=0)


def test_limit() -> None:
    fused = reciprocal_rank_fusion({"semantic": scored((X, 0.9), (Y, 0.8), (Z, 0.7))}, limit=2)

    assert [c.chunk.path for c in fused] == ["x.py", "y.py"]
