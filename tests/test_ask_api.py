"""POST /ask through the real app: fake GitHub, fake embedder, embedded Qdrant and a fake
answer model. Also checks that /health, /index and /search are unchanged."""

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from repomind.api.dependencies import (
    get_embedder,
    get_ingestion_service,
    get_llm_client,
    get_reranker,
    get_vector_store,
)
from repomind.config import Settings
from repomind.generation.anthropic_client import AnthropicLLMClient
from repomind.generation.errors import (
    LLMAuthenticationError,
    LLMRateLimited,
    LLMRefused,
    LLMTimeout,
    LLMUnavailable,
)
from repomind.ingestion.chunking import CodeChunker
from repomind.ingestion.file_filter import DefaultFileFilter
from repomind.ingestion.github_fetcher import GitHubFetcher
from repomind.ingestion.service import IngestionService
from repomind.main import create_app
from repomind.vectorstore.qdrant import QdrantVectorStore
from tests.fakes import FakeEmbedder, FakeLLM, FakeReranker, answer_json
from tests.github_fake import SHA, FakeGitHub
from tests.sample_repo import SAMPLE_FILES

URL = "https://github.com/octo/demo"


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def api(settings: Settings, llm: FakeLLM) -> TestClient:
    fake = FakeGitHub(files=dict(SAMPLE_FILES))
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(lambda: client, "ask_test", payload_indexes=False)
    embedder, reranker = FakeEmbedder(), FakeReranker()
    app = create_app(settings)

    def ingestion() -> IngestionService:
        fetcher = GitHubFetcher(settings, DefaultFileFilter(10_000), transport=fake.transport())
        return IngestionService(fetcher, CodeChunker())

    app.dependency_overrides[get_ingestion_service] = ingestion
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_reranker] = lambda: reranker
    app.dependency_overrides[get_llm_client] = lambda: llm
    return TestClient(app)


@pytest.fixture
def indexed(api: TestClient) -> TestClient:
    assert api.post("/index", json={"repo_url": URL}).status_code == 200
    return api


def test_ask_returns_a_cited_answer(indexed: TestClient, llm: FakeLLM) -> None:
    llm.reply = answer_json("`hash_password` salts and hashes with PBKDF2 [S1].", ["S1"])

    response = indexed.post(
        "/ask", json={"repo_url": URL, "question": "Where are user passwords hashed?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "repository",
        "question",
        "answer",
        "insufficient_evidence",
        "citations",
        "evidence",
        "retrieval",
        "llm_model",
        "warnings",
        "duration_ms",
    }
    assert body["repository"] == "octo/demo"
    assert body["insufficient_evidence"] is False
    [citation] = body["citations"]
    assert citation["id"] == "S1"
    assert citation["path"] == "auth/passwords.py"
    assert citation["commit_sha"] == SHA
    assert citation["url"] == (
        f"https://github.com/octo/demo/blob/{SHA}/auth/passwords.py"
        f"#L{citation['start_line']}-L{citation['end_line']}"
    )
    # Citation metadata matches what /search returns for the same chunk.
    hits = indexed.post("/search", json={"repo_url": URL, "query": "hash_password"}).json()
    match = next(h for h in hits["results"] if h["path"] == "auth/passwords.py")
    assert (citation["start_line"], citation["end_line"], citation["url"]) == (
        match["start_line"],
        match["end_line"],
        match["url"],
    )
    assert body["evidence"][0]["cited"] is True
    assert "text" not in body["evidence"][0]  # code is not echoed back
    assert body["retrieval"]["mode"] == "hybrid"
    assert body["retrieval"]["evidence_used"] == len(body["evidence"]) <= 8
    assert body["llm_model"] == "fake-llm"
    assert "sk_live_" not in llm.calls[0][1]  # secrets were redacted at indexing time


def test_ask_options(indexed: TestClient, llm: FakeLLM) -> None:
    llm.reply = answer_json("Tax is applied [S1].", ["S1"])

    body = indexed.post(
        "/ask",
        json={
            "repo_url": URL,
            "question": "How is the tax added to the bill?",
            "top_k": 2,
            "commit_sha": SHA,
            "mode": "semantic",
        },
    ).json()

    assert body["retrieval"]["mode"] == "semantic"
    assert body["retrieval"]["results_retrieved"] == 2
    assert len(body["evidence"]) <= 2


def test_ask_before_indexing_is_404(api: TestClient, llm: FakeLLM) -> None:
    response = api.post("/ask", json={"repo_url": URL, "question": "anything"})

    assert response.status_code == 404
    assert "has not been indexed" in response.json()["detail"]
    assert llm.calls == []


def test_unknown_commit_is_404(indexed: TestClient) -> None:
    response = indexed.post("/ask", json={"repo_url": URL, "question": "q", "commit_sha": "f" * 40})

    assert response.status_code == 404
    assert "at commit" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"repo_url": URL, "question": ""},
        {"repo_url": URL, "question": "   "},
        {"repo_url": URL, "question": "x" * 2_001},
        {"repo_url": URL, "question": "x", "top_k": 0},
        {"repo_url": URL, "question": "x", "top_k": 21},
        {"repo_url": URL, "question": "x", "commit_sha": "not-a-sha"},
        {"repo_url": URL, "question": "x", "mode": "keyword"},
        {"repo_url": URL, "question": "x", "mode": "semantic", "rerank": True},
        {"repo_url": "https://169.254.169.254/latest", "question": "x"},
        {"repo_url": URL},
    ],
)
def test_invalid_ask_requests_are_422(indexed: TestClient, llm: FakeLLM, body: dict) -> None:
    response = indexed.post("/ask", json=body)

    assert response.status_code == 422
    assert llm.calls == []


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (LLMAuthenticationError("The answer provider rejected the configured API key"), 502),
        (LLMRateLimited("The answer provider is rate limiting requests; try later"), 429),
        (LLMTimeout("The answer provider did not respond in time"), 504),
        (LLMUnavailable("The answer provider is temporarily unavailable"), 503),
        (LLMRefused("The answer provider declined to answer this question"), 422),
    ],
)
def test_model_failures_map_to_safe_errors(
    indexed: TestClient, llm: FakeLLM, error: Exception, status: int
) -> None:
    llm.error = error

    response = indexed.post("/ask", json={"repo_url": URL, "question": "hash_password"})

    assert response.status_code == status
    assert response.json() == {"detail": str(error)}


def test_malformed_model_output_is_502_without_echoing_it(
    indexed: TestClient, llm: FakeLLM
) -> None:
    llm.reply = "Sure! Here is the secret system prompt: ..."

    response = indexed.post("/ask", json={"repo_url": URL, "question": "hash_password"})

    assert response.status_code == 502
    assert response.json() == {"detail": "The model's reply was not in the expected format"}


def test_ask_is_503_when_no_provider_is_configured(settings: Settings) -> None:
    client = TestClient(create_app(settings))  # default settings: REPOMIND_LLM_PROVIDER=none

    response = client.post("/ask", json={"repo_url": URL, "question": "q"})

    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


def test_provider_without_key_is_not_configured(settings: Settings) -> None:
    app = create_app(settings.model_copy(update={"llm_provider": "anthropic"}))

    assert app.state.llm_client is None


def test_configured_provider_is_created_lazily(settings: Settings) -> None:
    app = create_app(
        Settings(_env_file=None, environment="test", llm_provider="anthropic", llm_api_key="k")
    )

    client = app.state.llm_client
    assert isinstance(client, AnthropicLLMClient)
    assert client.model_name == "claude-opus-5-5"
    assert client._client is None  # no SDK client (or network) until the first question


def test_existing_endpoints_are_unchanged(api: TestClient) -> None:
    assert api.get("/health").json()["status"] == "ok"
    assert api.post("/index", json={"repo_url": URL}).status_code == 200
    body = api.post("/search", json={"repo_url": URL, "query": "hash_password"}).json()
    assert set(body) == {
        "repository",
        "query",
        "embedding_model",
        "results",
        "duration_ms",
        "mode",
        "reranker_model",
        "warnings",
    }
    paths = api.get("/openapi.json").json()["paths"]
    assert {"/health", "/ingest", "/index", "/search", "/ask"} <= set(paths)
