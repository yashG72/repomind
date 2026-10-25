"""The retrieval evaluation harness, run with deterministic fakes (FakeEmbedder, embedded
Qdrant, FakeReranker). These tests check the metrics and that isolation holds in every mode;
they do not assert that one mode beats another. Print the full report with
`uv run python -m tests.evaluation` (add --real for the real models)."""

import uuid

import pytest
from qdrant_client import QdrantClient

from repomind.retrieval.search import SearchHit
from repomind.vectorstore.qdrant import QdrantVectorStore
from tests.evaluation.dataset import QUERIES, EvalQuery
from tests.evaluation.harness import MODES, ModeResult, build_index, evaluate, format_report, judge
from tests.fakes import FakeEmbedder, FakeReranker

QUERY = EvalQuery(
    "q", "test", "q", (("a.py", "def a"), ("b.py", None)), repo_url="https://github.com/o/r"
)


def hit(path: str, symbol: str | None = None, repo: str = "o/r") -> SearchHit:
    return SearchHit(
        score=0.0,
        path=path,
        start_line=1,
        end_line=1,
        symbol=symbol,
        language=None,
        commit_sha="a" * 40,
        url=f"https://github.com/{repo}/blob/{'a' * 40}/{path}#L1-L1",
        text="",
    )


def test_judge_ranks_recall_and_leaks() -> None:
    hits = [hit("x.py"), hit("a.py", "def other"), hit("b.py", "def b"), hit("a.py", "def a")]

    result = judge(QUERY, hits)

    assert result.first_relevant_rank == 3  # b.py matches any symbol; a.py needs "def a"
    assert result.relevant_found == {1: 0, 3: 1, 5: 2}
    assert result.leaks == 0
    assert judge(QUERY, [hit("a.py", "def a", repo="o/other")]).leaks == 1


def test_mode_metrics() -> None:
    mode = ModeResult("m")
    mode.queries = [
        judge(QUERY, [hit("a.py", "def a"), hit("b.py")]),  # rank 1, recall@1 = 1/2
        judge(QUERY, [hit("x.py"), hit("y.py"), hit("b.py")]),  # rank 3, recall@3 = 1/2
        judge(EvalQuery("n", "no match", "q", ()), [hit("x.py")]),  # not judged
    ]

    assert mode.recall(1) == pytest.approx((0.5 + 0) / 2)
    assert mode.recall(3) == pytest.approx((1.0 + 0.5) / 2)
    assert mode.mrr == pytest.approx((1 + 1 / 3) / 2)


@pytest.fixture(scope="module")
def results() -> list[ModeResult]:
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(lambda: client, f"eval_{uuid.uuid4().hex[:8]}", payload_indexes=False)
    embedder = FakeEmbedder()
    build_index(embedder, store)
    return evaluate(embedder, store, FakeReranker())


def test_every_mode_answers_every_question_without_leaks(results) -> None:
    assert [r.name for r in results] == list(MODES)
    for mode in results:
        assert len(mode.queries) == len(QUERIES)
        assert mode.leaks == 0, mode.name
        assert 0.0 <= mode.mrr <= 1.0


def test_evaluation_is_deterministic(results) -> None:
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(lambda: client, "again", payload_indexes=False)
    embedder = FakeEmbedder()
    build_index(embedder, store)

    again = evaluate(embedder, store, FakeReranker())

    assert format_report(again, "t") == format_report(results, "t")


def test_dataset_covers_every_required_category() -> None:
    categories = {q.category for q in QUERIES}

    assert categories >= {
        "exact function",
        "exact identifier",
        "conceptual",
        "keyword vs semantic",
        "repository isolation",
        "no match",
    }
    assert len({q.id for q in QUERIES}) == len(QUERIES)
