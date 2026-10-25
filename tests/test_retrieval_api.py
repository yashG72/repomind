"""POST /index and POST /search through the real app, with a fake embedder, embedded Qdrant
and a fake GitHub."""

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from repomind.api.dependencies import (
    get_embedder,
    get_ingestion_service,
    get_reranker,
    get_vector_store,
)
from repomind.config import Settings
from repomind.ingestion.chunking import CodeChunker
from repomind.ingestion.file_filter import DefaultFileFilter
from repomind.ingestion.github_fetcher import GitHubFetcher
from repomind.ingestion.service import IngestionService
from repomind.main import create_app
from repomind.vectorstore.qdrant import QdrantVectorStore
from tests.fakes import FakeEmbedder, FakeReranker
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
def reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture
def api(
    settings: Settings, embedder: FakeEmbedder, store: QdrantVectorStore, reranker: FakeReranker
) -> TestClient:
    fake = FakeGitHub(files=dict(SAMPLE_FILES))
    app = create_app(settings)

    def ingestion() -> IngestionService:
        fetcher = GitHubFetcher(settings, DefaultFileFilter(10_000), transport=fake.transport())
        return IngestionService(fetcher, CodeChunker())

    app.dependency_overrides[get_ingestion_service] = ingestion
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_reranker] = lambda: reranker
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


# --- Phase 4: hybrid search and reranking ---------------------------------------------------


def test_default_search_is_hybrid_and_reranked(api: TestClient, reranker: FakeReranker) -> None:
    api.post("/index", json={"repo_url": URL})

    result = api.post("/search", json={"repo_url": URL, "query": "hash_password"}).json()

    assert result["mode"] == "hybrid"
    assert result["reranker_model"] == "fake-reranker"
    assert result["warnings"] == []
    top = result["results"][0]
    assert (top["path"], top["symbol"]) == ("auth/passwords.py", "def hash_password")
    assert top["scores"]["keyword_rank"] == 1
    assert top["scores"]["rerank"] == top["score"]
    # Phase 3 fields and citation format are unchanged.
    assert set(top) >= {"score", "path", "start_line", "end_line", "symbol", "language"}
    assert top["url"].endswith(f"#L{top['start_line']}-L{top['end_line']}")
    assert len(reranker.calls) == 1 and 5 <= reranker.calls[0] <= 20  # bounded candidate set


def test_semantic_mode_matches_phase3(api: TestClient, reranker: FakeReranker) -> None:
    api.post("/index", json={"repo_url": URL})

    result = api.post(
        "/search", json={"repo_url": URL, "query": "hash password salt", "mode": "semantic"}
    ).json()

    assert result["mode"] == "semantic" and result["reranker_model"] is None
    top = result["results"][0]
    assert top["path"] == "auth/passwords.py"
    assert top["score"] == pytest.approx(top["scores"]["semantic"], abs=1e-4)
    assert top["scores"]["keyword"] is None and top["scores"]["rrf"] is None
    assert reranker.calls == []


def test_hybrid_without_rerank(api: TestClient, reranker: FakeReranker) -> None:
    api.post("/index", json={"repo_url": URL})

    result = api.post(
        "/search", json={"repo_url": URL, "query": "retry_with_backoff", "rerank": False}
    ).json()

    assert result["reranker_model"] is None and reranker.calls == []
    assert result["results"][0]["symbol"] == "def retry_with_backoff"
    assert result["results"][0]["score"] == result["results"][0]["scores"]["rrf"]


@pytest.mark.parametrize(
    "extra",
    [{"mode": "keyword"}, {"mode": "semantic", "rerank": True}, {"rerank": "maybe"}],
)
def test_invalid_mode_or_rerank_is_422(api: TestClient, extra: dict) -> None:
    api.post("/index", json={"repo_url": URL})

    response = api.post("/search", json={"repo_url": URL, "query": "x", **extra})

    assert response.status_code == 422


def test_reranker_failure_is_503_without_internals(api: TestClient, reranker: FakeReranker) -> None:
    api.post("/index", json={"repo_url": URL})
    reranker.fail = True

    response = api.post("/search", json={"repo_url": URL, "query": "password"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Could not load reranking model 'fake-reranker'"}
    assert "Traceback" not in response.text
    # Semantic mode does not need the reranker and keeps working.
    ok = api.post("/search", json={"repo_url": URL, "query": "password", "mode": "semantic"})
    assert ok.status_code == 200


def test_rerank_disabled_by_settings(
    settings: Settings, embedder, store, reranker: FakeReranker
) -> None:
    app = create_app(settings.model_copy(update={"rerank_enabled": False}))
    fake = FakeGitHub(files=dict(SAMPLE_FILES))
    app.dependency_overrides[get_ingestion_service] = lambda: IngestionService(
        GitHubFetcher(settings, DefaultFileFilter(10_000), transport=fake.transport()),
        CodeChunker(),
    )
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_reranker] = lambda: reranker
    client = TestClient(app)
    client.post("/index", json={"repo_url": URL})

    default = client.post("/search", json={"repo_url": URL, "query": "password"})
    forced = client.post("/search", json={"repo_url": URL, "query": "password", "rerank": True})

    assert default.status_code == 200 and default.json()["reranker_model"] is None
    assert forced.status_code == 422 and "disabled" in forced.json()["detail"]
    assert reranker.calls == []


def test_phase3_collection_still_searches_with_a_warning(
    settings: Settings, embedder: FakeEmbedder, reranker: FakeReranker
) -> None:
    client = QdrantClient(":memory:")
    client.create_collection(  # how Phase 3 created collections: no keyword index
        "legacy", vectors_config=models.VectorParams(size=64, distance=models.Distance.COSINE)
    )
    legacy = QdrantVectorStore(lambda: client, "legacy", payload_indexes=False)
    app = create_app(settings)
    fake = FakeGitHub(files=dict(SAMPLE_FILES))
    app.dependency_overrides[get_ingestion_service] = lambda: IngestionService(
        GitHubFetcher(settings, DefaultFileFilter(10_000), transport=fake.transport()),
        CodeChunker(),
    )
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_vector_store] = lambda: legacy
    app.dependency_overrides[get_reranker] = lambda: reranker
    api = TestClient(app)
    assert api.post("/index", json={"repo_url": URL}).status_code == 200

    hybrid = api.post("/search", json={"repo_url": URL, "query": "hash password salt"})
    semantic = api.post(
        "/search", json={"repo_url": URL, "query": "hash password salt", "mode": "semantic"}
    )

    assert hybrid.status_code == semantic.status_code == 200
    assert "Keyword search is unavailable" in hybrid.json()["warnings"][0]
    assert hybrid.json()["results"][0]["path"] == "auth/passwords.py"
    assert hybrid.json()["results"][0]["scores"]["keyword"] is None
    assert semantic.json()["warnings"] == []
