"""CrossEncoderReranker with a fake sentence_transformers module: no model download, no torch.

These tests check loading, validation and error handling. Whether the real model ranks well
is checked separately in tests/integration/test_real_reranker.py.
"""

import math
import sys
import threading
import types
import uuid

import numpy as np
import pytest

from repomind.core.models import CodeChunk, RepoRef, ScoredChunk
from repomind.reranking.cross_encoder import CrossEncoderReranker, order_by_scores
from repomind.reranking.errors import RerankerError

REPO = RepoRef(owner="octo", name="demo")


class FakeCrossEncoder:
    """Mimics the parts of sentence_transformers.CrossEncoder that we use."""

    loads = 0
    last_kwargs: dict = {}

    def __init__(self, name, **kwargs):
        FakeCrossEncoder.loads += 1
        FakeCrossEncoder.last_kwargs = kwargs
        self.output: object = None
        self.pairs: list = []

    def predict(self, pairs, **kwargs):
        self.pairs = list(pairs)
        if self.output is not None:
            return self.output
        # Score = how often the query's first word appears in the document.
        return np.array([doc.count(query.split()[0]) for query, doc in pairs], dtype=float)


@pytest.fixture
def fake_library(monkeypatch: pytest.MonkeyPatch):
    FakeCrossEncoder.loads = 0
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


def candidate(text: str, score: float = 0.0) -> ScoredChunk:
    return ScoredChunk(
        chunk=CodeChunk(
            chunk_id=uuid.uuid5(uuid.NAMESPACE_URL, text).hex,
            repo=REPO,
            commit_sha="a" * 40,
            path=f"{text[:8]}.py",
            start_line=1,
            end_line=1,
            text=text,
        ),
        score=score,
    )


def test_model_is_loaded_lazily_and_only_once(fake_library) -> None:
    reranker = CrossEncoderReranker("some/model")
    assert FakeCrossEncoder.loads == 0 and not reranker.is_loaded

    reranker.rerank("retry", [candidate("retry retry"), candidate("other")])
    reranker.rerank("retry", [candidate("retry")])

    assert FakeCrossEncoder.loads == 1 and reranker.is_loaded
    assert FakeCrossEncoder.last_kwargs["trust_remote_code"] is False


def test_empty_candidates_do_not_load_the_model(fake_library) -> None:
    assert CrossEncoderReranker("some/model").rerank("q", []) == []
    assert FakeCrossEncoder.loads == 0


def test_concurrent_first_use_loads_once(fake_library) -> None:
    reranker = CrossEncoderReranker("some/model")
    threads = [
        threading.Thread(target=reranker.rerank, args=("x", [candidate("x")])) for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert FakeCrossEncoder.loads == 1


def test_candidates_are_reordered_with_reranker_scores(fake_library) -> None:
    low, high, mid = (
        candidate("retry", 0.9),
        candidate("retry retry retry", 0.1),
        candidate("retry retry", 0.5),
    )

    result = CrossEncoderReranker("m").rerank("retry now", [low, high, mid])

    assert [r.chunk for r in result] == [high.chunk, mid.chunk, low.chunk]
    # The model's scores replace the incoming ones (each document also names its path).
    assert [r.score for r in result] == [4.0, 3.0, 2.0]


def test_model_sees_path_symbol_and_code_but_bounded(fake_library) -> None:
    reranker = CrossEncoderReranker("m")
    long_text = "x" * 10_000

    reranker.rerank("q", [candidate(long_text)])

    [(query, document)] = reranker._model.pairs
    assert query == "q"
    assert document.startswith("xxxxxxxx.py\n\n")
    assert len(document) == 4_000


def test_too_many_candidates_are_refused(fake_library) -> None:
    reranker = CrossEncoderReranker("m", max_candidates=2)

    with pytest.raises(ValueError, match="at most 2"):
        reranker.rerank("q", [candidate("a"), candidate("b"), candidate("c")])


def test_load_failure_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("sentence_transformers")

    def broken(*args, **kwargs):
        raise OSError("We couldn't connect to 'https://huggingface.co' to load the files")

    module.CrossEncoder = broken
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    with pytest.raises(RerankerError, match="Could not load reranking model 'missing/model'"):
        CrossEncoderReranker("missing/model").rerank("q", [candidate("a")])


@pytest.mark.parametrize(
    "output",
    [
        np.array([1.0]),  # too few scores
        np.array([1.0, 2.0, 3.0]),  # too many
        np.array([1.0, math.nan]),
        np.array([math.inf, 1.0]),
    ],
)
def test_invalid_model_output_is_rejected(fake_library, output) -> None:
    reranker = CrossEncoderReranker("m")
    reranker._load()
    reranker._model.output = output

    with pytest.raises(RerankerError):
        reranker.rerank("q", [candidate("a"), candidate("b")])


def test_prediction_failure_is_a_clear_error(fake_library) -> None:
    reranker = CrossEncoderReranker("m")
    reranker._load()

    def broken(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    reranker._model.predict = broken

    with pytest.raises(RerankerError, match="failed to score"):
        reranker.rerank("q", [candidate("a")])


def test_order_by_scores_keeps_input_order_for_ties() -> None:
    a, b, c = candidate("a"), candidate("b"), candidate("c")

    result = order_by_scores([a, b, c], [1.0, 2.0, 1.0])

    assert [r.chunk for r in result] == [b.chunk, a.chunk, c.chunk]
