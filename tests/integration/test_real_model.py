"""End-to-end retrieval with the real embedding model and a real Qdrant.

Skipped unless REPOMIND_TEST_EMBEDDING_MODEL is set to a model id or local model folder:

    REPOMIND_TEST_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \\
    REPOMIND_TEST_QDRANT_URL=http://localhost:6333 \\
    uv run pytest -m integration

Without REPOMIND_TEST_QDRANT_URL the test uses Qdrant local mode in a temporary folder.
GitHub is still simulated, so the expected answers are known.
"""

import os
import uuid

import pytest
from qdrant_client import QdrantClient

from repomind.config import Settings
from repomind.core.models import RepoRef
from repomind.embeddings.sentence_transformer import SentenceTransformerEmbedder
from repomind.ingestion.chunking import CodeChunker
from repomind.ingestion.file_filter import DefaultFileFilter
from repomind.ingestion.github_fetcher import GitHubFetcher
from repomind.ingestion.service import IngestionService
from repomind.retrieval.indexing import IndexingService
from repomind.retrieval.search import SearchService
from repomind.vectorstore.qdrant import QdrantVectorStore
from tests.github_fake import FakeGitHub
from tests.sample_repo import QUESTIONS, SAMPLE_FILES

MODEL = os.environ.get("REPOMIND_TEST_EMBEDDING_MODEL")
QDRANT_URL = os.environ.get("REPOMIND_TEST_QDRANT_URL")
URL = "https://github.com/octo/demo"
REPO = RepoRef(owner="octo", name="demo")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not MODEL, reason="set REPOMIND_TEST_EMBEDDING_MODEL to run"),
]


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    assert MODEL
    return SentenceTransformerEmbedder(MODEL)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        qdrant_url=QDRANT_URL,
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection=f"it_{uuid.uuid4().hex[:12]}",
    )


@pytest.fixture
def store(settings: Settings):
    vector_store = QdrantVectorStore.from_settings(settings)
    yield vector_store
    if QDRANT_URL:
        QdrantClient(url=QDRANT_URL).delete_collection(settings.qdrant_collection)
    vector_store.close()


def index(settings: Settings, embedder, store) -> IndexingService:
    fake = FakeGitHub(files=dict(SAMPLE_FILES))
    fetcher = GitHubFetcher(settings, DefaultFileFilter(10_000), transport=fake.transport())
    service = IndexingService(
        IngestionService(fetcher, CodeChunker()), embedder, store, settings.qdrant_collection
    )
    return service


def test_dimension_is_read_from_the_model(embedder) -> None:
    vector = embedder.embed_query("hello")

    assert len(vector) == embedder.dimension
    assert sum(x * x for x in vector) == pytest.approx(1.0, abs=1e-4)  # normalized


def test_natural_language_questions_find_the_right_code(settings, embedder, store) -> None:
    summary = index(settings, embedder, store).index(URL)
    assert summary.embeddings_computed == summary.chunks_indexed > 0

    search = SearchService(embedder, store)
    for question, (path, symbol) in QUESTIONS.items():
        top = search.search(URL, question, top_k=3).results[0]
        assert (top.path, top.symbol) == (path, symbol), question


def test_vectors_persist_and_reindexing_reuses_them(settings, embedder, store) -> None:
    service = index(settings, embedder, store)
    first = service.index(URL)
    store.close()  # simulate a restart: a brand-new client on the same storage

    reopened = QdrantVectorStore.from_settings(settings)
    try:
        assert reopened.count(REPO) == first.chunks_indexed
        second = index(settings, embedder, reopened).index(URL)
        assert second.embeddings_computed == 0
        assert second.embeddings_reused == first.chunks_indexed
        assert reopened.count(REPO) == first.chunks_indexed  # no dupes
    finally:
        reopened.close()
