"""POST /ask's pipeline with the real Anthropic API. These tests make paid API calls.

Skipped unless REPOMIND_TEST_LLM_API_KEY is set (the normal suite never calls a provider):

    REPOMIND_TEST_LLM_API_KEY=sk-ant-... \\
    uv run pytest -m integration tests/integration/test_real_llm.py

REPOMIND_TEST_LLM_MODEL picks the model (default: the app's default). Retrieval uses the fake
embedder and an in-memory store over the sample repository, so the right answers are known.
"""

import os

import pytest
from pydantic import SecretStr

from repomind.core.models import RepoRef, SourceFile
from repomind.generation.answer import AnswerService
from repomind.generation.anthropic_client import AnthropicLLMClient
from repomind.generation.context import ContextBuilder
from repomind.generation.errors import LLMAuthenticationError
from repomind.ingestion.chunking import CodeChunker
from repomind.retrieval.search import SearchService
from tests.fakes import FakeEmbedder, InMemoryVectorStore
from tests.sample_repo import SAMPLE_FILES

API_KEY = os.environ.get("REPOMIND_TEST_LLM_API_KEY")
MODEL = os.environ.get("REPOMIND_TEST_LLM_MODEL", "claude-opus-5-5")
REPO = RepoRef(owner="octo", name="demo")
URL = "https://github.com/octo/demo"
SHA = "c" * 40

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not API_KEY, reason="set REPOMIND_TEST_LLM_API_KEY to run"),
]


@pytest.fixture(scope="module")
def service() -> AnswerService:
    assert API_KEY
    embedder, store = FakeEmbedder(), InMemoryVectorStore()
    chunker = CodeChunker()
    chunks = [
        chunk
        for path, data in SAMPLE_FILES.items()
        if path.endswith(".py") and path != "config.py"
        for chunk in chunker.chunk(
            REPO, SHA, SourceFile(path=path, content=data.decode(), size_bytes=len(data))
        )
    ]
    store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]), embedder.model_name)
    llm = AnthropicLLMClient(SecretStr(API_KEY), MODEL, effort="low")
    return AnswerService(SearchService(embedder, store), llm, ContextBuilder(5, 10_000))


def test_answer_cites_the_right_file(service: AnswerService) -> None:
    response = service.ask(URL, "Where are user passwords hashed, and how?", top_k=5)

    assert response.insufficient_evidence is False
    assert "auth/passwords.py" in {c.path for c in response.citations}
    assert all(c.commit_sha == SHA for c in response.citations)


def test_missing_feature_is_reported_as_insufficient(service: AnswerService) -> None:
    response = service.ask(URL, "How does this repository send email notifications?", top_k=5)

    assert response.insufficient_evidence is True


def test_bad_key_is_a_safe_error() -> None:
    llm = AnthropicLLMClient(SecretStr("sk-ant-invalid"), MODEL, max_retries=0)

    with pytest.raises(LLMAuthenticationError):
        llm.generate("Reply with {}", "hi", {"type": "object"})
