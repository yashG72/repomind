"""POST /index and POST /search through the real app, with a fake embedder, embedded Qdrant
and a fake GitHub."""

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from repomind.api.dependencies import get_embedder, get_ingestion_service, get_vector_store
from repomind.config import Settings
from repomind.ingestion.chunking import CodeChunker
from repomind.ingestion.file_filter import DefaultFileFilter
from repomind.ingestion.github_fetcher import GitHubFetcher
from repomind.ingestion.service import IngestionService
from repomind.main import create_app
from repomind.vectorstore.qdrant import QdrantVectorStore
from tests.fakes import FakeEmbedder
from tests.github_fake import SHA, FakeGitHub
from tests.sample_repo import SAMPLE_FILES

URL = "https://github.com/octo/demo"


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store() -> QdrantVectorStore:
    client = QdrantClient(":memory:")
    return QdrantVectorStore(lambda: client, "api_test", payload_indexes=False)


@pytest.fixture
def api(settings: Settings, embedder: FakeEmbedder, store: QdrantVectorStore) -> TestClient:
    fake = FakeGitHub(files=dict(SAMPLE_FILES))
    app = create_app(settings)

    def ingestion() -> IngestionService:
        fetcher = GitHubFetcher(settings, DefaultFileFilter(10_000), transport=fake.transport())
        return IngestionService(fetcher, CodeChunker())

    app.dependency_overrides[get_ingestion_service] = ingestion
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_vector_store] = lambda: store
    return TestClient(app)


def test_index_then_search(api: TestClient) -> None:
    indexed = api.post("/index", json={"repo_url": URL})

    assert indexed.status_code == 200
    body = indexed.json()
    assert body["chunks_indexed"] == body["ingestion"]["chunks_created"] > 0
    assert body["embeddings_computed"] == body["chunks_indexed"]
    assert body["embedding_dimension"] == 64

    found = api.post("/search", json={"repo_url": URL, "query": "hash password salt"})

    assert found.status_code == 200
    result = found.json()
    assert result["repository"] == "octo/demo"
    top = result["results"][0]
    assert top["path"] == "auth/passwords.py"
    assert top["symbol"] == "def hash_password"
    assert top["commit_sha"] == SHA
    assert top["url"] == (
        f"https://github.com/octo/demo/blob/{SHA}/auth/passwords.py"
        f"#L{top['start_line']}-L{top['end_line']}"
    )
    assert "def hash_password(password):" in top["text"]
    assert len(result["results"]) == 5  # default top_k


def test_reindex_reports_reuse(api: TestClient) -> None:
    api.post("/index", json={"repo_url": URL})

    second = api.post("/index", json={"repo_url": URL}).json()

    assert second["embeddings_computed"] == 0
    assert second["embeddings_reused"] == second["chunks_indexed"]


def test_search_before_indexing_is_404(api: TestClient) -> None:
    response = api.post("/search", json={"repo_url": URL, "query": "anything"})

    assert response.status_code == 404
    assert "has not been indexed" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"repo_url": URL, "query": ""},
        {"repo_url": URL, "query": "x", "top_k": 0},
        {"repo_url": URL, "query": "x", "top_k": 51},
        {"repo_url": URL, "query": "x", "commit_sha": "not-a-sha"},
        {"repo_url": URL, "query": "x" * 2_001},
        {"repo_url": "https://169.254.169.254/latest", "query": "x"},
        {"query": "x"},
    ],
)
def test_invalid_search_requests_are_422(api: TestClient, body: dict) -> None:
    assert api.post("/search", json=body).status_code == 422


def test_blank_query_is_422(api: TestClient) -> None:
    api.post("/index", json={"repo_url": URL})

    assert api.post("/search", json={"repo_url": URL, "query": "   "}).status_code == 422


def test_embedding_failure_is_503(api: TestClient, embedder: FakeEmbedder) -> None:
    embedder.fail = True

    response = api.post("/index", json={"repo_url": URL})

    assert response.status_code == 503
    assert "embedding model" in response.json()["detail"]


def test_unavailable_vector_store_is_503(api: TestClient, store: QdrantVectorStore) -> None:
    def broken():
        raise ConnectionRefusedError("qdrant is down")

    store._client_factory = broken
    store._client = None

    assert api.post("/index", json={"repo_url": URL}).status_code == 503
    assert api.post("/search", json={"repo_url": URL, "query": "x"}).status_code == 503


def test_secret_values_are_not_returned_or_stored(api: TestClient, store) -> None:
    api.post("/index", json={"repo_url": URL})

    response = api.post(
        "/search", json={"repo_url": URL, "query": "stripe key config", "top_k": 50}
    )

    assert "sk_live_" not in response.text
    assert "[REDACTED]" in response.text
