"""The retrieval pipeline (SearchService): semantic-only mode, hybrid mode with RRF, and
reranking. A scripted store returns fixed result lists so each stage can be checked exactly."""

import uuid
from collections.abc import Sequence

import pytest

from repomind.core.models import CodeChunk, RepoRef, ScoredChunk
from repomind.reranking.errors import RerankerError
from repomind.retrieval.errors import InvalidQuery, RepositoryNotIndexed
from repomind.retrieval.search import (
    KEYWORD_UNAVAILABLE_WARNING,
    SearchConfig,
    SearchMode,
    SearchService,
)
from tests.fakes import FakeEmbedder, FakeReranker

URL = "https://github.com/octo/demo"
REPO = RepoRef(owner="octo", name="demo")
OTHER = RepoRef(owner="octo", name="other")
SHA1, SHA2 = "1" * 40, "2" * 40


def chunk(path: str, repo: RepoRef = REPO, sha: str = SHA1, text: str = "code") -> CodeChunk:
    return CodeChunk(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_URL, f"{repo.full_name}{sha}{path}").hex,
        repo=repo,
        commit_sha=sha,
        path=path,
        start_line=1,
        end_line=2,
        symbol=None,
        text=text,
    )


class ScriptedStore:
    """Returns preset semantic and keyword lists and records what it was asked."""

    def __init__(self, semantic: list[ScoredChunk], keyword: list[ScoredChunk], keywords=True):
        self.semantic = semantic
        self.keyword = keyword
        self.keywords = keywords
        self.calls: list[tuple] = []

    def count(self, repo: RepoRef, commit_sha: str | None = None) -> int:
        return 0 if repo.name == "missing" else 10

    def search(self, repo, vector: Sequence[float], limit, embedding_model, commit_sha=None):
        self.calls.append(("semantic", repo, limit, commit_sha))
        return self.semantic[:limit]

    def supports_keyword_search(self) -> bool:
        return self.keywords

    def keyword_search(self, repo, query, limit, commit_sha=None):
        self.calls.append(("keyword", repo, limit, commit_sha))
        return self.keyword[:limit]


def scored(*items: tuple[CodeChunk, float]) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=c, score=s) for c, s in items]


A, B, C, D = chunk("a.py"), chunk("b.py"), chunk("c.py"), chunk("d.py")


def service(store, reranker=None, **config) -> SearchService:
    return SearchService(FakeEmbedder(), store, reranker, SearchConfig(**config))


def test_semantic_mode_is_phase3_behaviour() -> None:
    store = ScriptedStore(scored((A, 0.91234), (B, 0.5)), scored((C, 9.0)))
    reranker = FakeReranker()

    response = service(store, reranker).search(URL, "q", top_k=5, mode=SearchMode.SEMANTIC)

    assert [h.path for h in response.results] == ["a.py", "b.py"]
    assert response.results[0].score == 0.9123  # cosine, rounded as in Phase 3
    assert response.results[0].scores.semantic_rank == 1
    assert response.results[0].scores.keyword is None
    assert response.mode == "semantic" and response.reranker_model is None
    assert store.calls == [("semantic", REPO, 5, None)]  # limit is top_k; no keyword search
    assert reranker.calls == []


def test_hybrid_fuses_both_lists_with_rrf() -> None:
    store = ScriptedStore(scored((A, 0.9), (B, 0.8)), scored((B, 7.0), (C, 3.0)))

    response = service(store, rerank_enabled=False).search(URL, "q", top_k=5)

    assert response.mode == "hybrid" and response.reranker_model is None
    assert [h.path for h in response.results] == ["b.py", "a.py", "c.py"]
    top = response.results[0]
    assert top.scores.semantic == 0.8 and top.scores.semantic_rank == 2
    assert top.scores.keyword == 7.0 and top.scores.keyword_rank == 1
    assert top.scores.rrf == pytest.approx(1 / 62 + 1 / 61, abs=1e-6)
    assert top.score == top.scores.rrf and top.scores.rerank is None


def test_hybrid_asks_each_retriever_for_its_candidate_count() -> None:
    store = ScriptedStore([], [])

    service(store, semantic_candidates=30, keyword_candidates=40).search(
        URL, "q", top_k=5, commit_sha=SHA1
    )

    assert store.calls == [("semantic", REPO, 30, SHA1), ("keyword", REPO, 40, SHA1)]


def test_candidate_counts_never_go_below_top_k() -> None:
    store = ScriptedStore([], [])

    service(store, semantic_candidates=3, keyword_candidates=3).search(URL, "q", top_k=10)

    assert [call[2] for call in store.calls] == [10, 10]


def test_query_is_embedded_once() -> None:
    embedder = FakeEmbedder()
    store = ScriptedStore(scored((A, 0.9)), scored((A, 1.0)))

    SearchService(embedder, store, FakeReranker()).search(URL, "q", top_k=5)

    assert embedder.queries_embedded == 1


def test_reranker_reorders_only_the_top_candidates() -> None:
    semantic = scored(*[(chunk(f"s{i}.py"), 1 - i / 100) for i in range(10)])
    store = ScriptedStore(semantic, [])
    reranker = FakeReranker()
    # Reverse the order of the 3 candidates the reranker sees.
    reranker.scores_override = [1.0, 2.0, 3.0]

    response = service(store, reranker, rerank_max_candidates=3).search(URL, "q", top_k=5)

    assert reranker.calls == [3]
    assert [h.path for h in response.results] == ["s2.py", "s1.py", "s0.py", "s3.py", "s4.py"]
    assert [h.scores.rerank for h in response.results] == [3.0, 2.0, 1.0, None, None]
    assert response.results[0].score == 3.0
    assert response.results[3].score == response.results[3].scores.rrf  # not reranked
    assert response.reranker_model == "fake-reranker"


def test_rerank_can_be_turned_off_per_request() -> None:
    store = ScriptedStore(scored((A, 0.9)), [])
    reranker = FakeReranker()

    response = service(store, reranker).search(URL, "q", top_k=5, rerank=False)

    assert reranker.calls == [] and response.reranker_model is None


def test_rerank_disabled_in_settings_is_skipped_or_refused() -> None:
    store = ScriptedStore(scored((A, 0.9)), [])
    reranker = FakeReranker()
    search = service(store, reranker, rerank_enabled=False)

    assert search.search(URL, "q", top_k=5).reranker_model is None
    with pytest.raises(InvalidQuery, match="disabled"):
        search.search(URL, "q", top_k=5, rerank=True)
    assert reranker.calls == []


def test_rerank_in_semantic_mode_is_refused() -> None:
    with pytest.raises(InvalidQuery, match="hybrid"):
        service(ScriptedStore([], []), FakeReranker()).search(
            URL, "q", top_k=5, mode=SearchMode.SEMANTIC, rerank=True
        )


def test_default_mode_comes_from_config() -> None:
    store = ScriptedStore(scored((A, 0.9)), scored((B, 1.0)))

    response = service(store, default_mode=SearchMode.SEMANTIC).search(URL, "q", top_k=5)

    assert response.mode == "semantic" and [h.path for h in response.results] == ["a.py"]


def test_no_reranker_configured_means_no_reranking() -> None:
    store = ScriptedStore(scored((A, 0.9)), [])

    assert service(store).search(URL, "q", top_k=5).reranker_model is None


def test_phase3_collection_falls_back_to_semantic_with_a_warning() -> None:
    store = ScriptedStore(scored((A, 0.9), (B, 0.8)), scored((C, 5.0)), keywords=False)

    response = service(store, FakeReranker()).search(URL, "q", top_k=5)

    assert response.warnings == [KEYWORD_UNAVAILABLE_WARNING]
    assert {h.path for h in response.results} == {"a.py", "b.py"}
    assert [c[0] for c in store.calls] == ["semantic"]


def test_out_of_scope_results_are_dropped_even_if_a_store_returns_them() -> None:
    wrong_repo, wrong_commit = chunk("x.py", repo=OTHER), chunk("y.py", sha=SHA2)
    store = ScriptedStore(scored((wrong_repo, 0.99), (A, 0.5)), scored((wrong_commit, 9.0)))

    response = service(store, rerank_enabled=False).search(URL, "q", top_k=5, commit_sha=SHA1)

    assert [h.path for h in response.results] == ["a.py"]


def test_reranker_returning_different_results_is_an_error() -> None:
    class Dropping(FakeReranker):
        def rerank(self, query, candidates):
            return list(candidates)[1:]

    store = ScriptedStore(scored((A, 0.9), (B, 0.8)), [])

    with pytest.raises(RerankerError, match="different set"):
        service(store, Dropping()).search(URL, "q", top_k=5)


def test_reranker_failure_propagates() -> None:
    reranker = FakeReranker()
    reranker.fail = True

    with pytest.raises(RerankerError):
        service(ScriptedStore(scored((A, 0.9)), []), reranker).search(URL, "q", top_k=5)


def test_nothing_found_skips_the_reranker() -> None:
    reranker = FakeReranker()

    response = service(ScriptedStore([], []), reranker).search(URL, "q", top_k=5)

    assert response.results == [] and reranker.calls == []


@pytest.mark.parametrize("mode", list(SearchMode))
def test_validation_happens_before_any_retrieval(mode) -> None:
    store = ScriptedStore([], [])

    with pytest.raises(InvalidQuery):
        service(store).search(URL, "   ", top_k=5, mode=mode)
    with pytest.raises(RepositoryNotIndexed):
        service(store).search("https://github.com/octo/missing", "q", top_k=5, mode=mode)
    assert store.calls == []


def test_top_k_limits_results() -> None:
    semantic = scored(*[(chunk(f"s{i}.py"), 1 - i / 100) for i in range(10)])

    response = service(ScriptedStore(semantic, []), FakeReranker()).search(URL, "q", top_k=3)

    assert len(response.results) == 3


def test_query_text_is_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    store = ScriptedStore(scored((A, 0.9)), scored((A, 1.0)))

    with caplog.at_level("INFO"):
        service(store, FakeReranker()).search(URL, "secret question text", top_k=5)

    record = next(r for r in caplog.records if r.getMessage() == "search completed")
    assert "secret question text" not in str(record.__dict__)
    assert record.query_chars == len("secret question text")
    assert record.mode == "hybrid" and record.reranked is True
