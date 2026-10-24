import sys
import threading
import types

import numpy as np
import pytest

from repomind.embeddings.errors import EmbeddingError
from repomind.embeddings.sentence_transformer import SentenceTransformerEmbedder


class FakeModel:
    """Mimics the parts of sentence_transformers.SentenceTransformer that we use."""

    loads = 0

    def __init__(self, name, device=None, cache_folder=None, local_files_only=False):
        FakeModel.loads += 1
        self.name = name
        self.output: object = None

    def get_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        if self.output is not None:
            return self.output
        return np.array([[1.0, 0.0, 0.0] for _ in texts])


@pytest.fixture
def fake_library(monkeypatch: pytest.MonkeyPatch):
    FakeModel.loads = 0
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


def test_model_is_loaded_lazily_and_only_once(fake_library) -> None:
    embedder = SentenceTransformerEmbedder("some/model")
    assert FakeModel.loads == 0 and not embedder.is_loaded

    embedder.embed_query("a")
    embedder.embed_documents(["b", "c"])
    _ = embedder.dimension

    assert FakeModel.loads == 1


def test_concurrent_first_use_loads_once(fake_library) -> None:
    embedder = SentenceTransformerEmbedder("some/model")
    threads = [threading.Thread(target=embedder.embed_query, args=("x",)) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert FakeModel.loads == 1


def test_dimension_comes_from_the_model(fake_library) -> None:
    assert SentenceTransformerEmbedder("some/model").dimension == 3


def test_vectors_are_plain_floats(fake_library) -> None:
    vectors = SentenceTransformerEmbedder("some/model").embed_documents(["a", "b"])

    assert vectors == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert all(type(x) is float for x in vectors[0])


def test_empty_input_does_not_load_the_model(fake_library) -> None:
    assert SentenceTransformerEmbedder("some/model").embed_documents([]) == []
    assert FakeModel.loads == 0


def test_load_failure_is_an_embedding_error(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("sentence_transformers")

    def broken(*args, **kwargs):
        raise OSError("model not found")

    module.SentenceTransformer = broken
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    with pytest.raises(EmbeddingError, match="Could not load embedding model"):
        SentenceTransformerEmbedder("missing/model").embed_query("x")


@pytest.mark.parametrize(
    "bad_output",
    [
        np.array([[float("nan"), 0.0, 0.0]]),  # NaN
        np.array([[0.0, 0.0, 0.0]]),  # all zeros
        np.array([[1.0, 0.0]]),  # wrong size
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),  # wrong count
    ],
)
def test_malformed_model_output_is_rejected(fake_library, bad_output) -> None:
    embedder = SentenceTransformerEmbedder("some/model")
    _ = embedder.dimension
    embedder._model.output = bad_output

    with pytest.raises(EmbeddingError):
        embedder.embed_query("x")


def test_encode_exception_is_an_embedding_error(fake_library) -> None:
    embedder = SentenceTransformerEmbedder("some/model")
    _ = embedder.dimension

    def explode(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    embedder._model.encode = explode

    with pytest.raises(EmbeddingError):
        embedder.embed_documents(["x"])
