"""IndexingService and SearchService with a fake embedder, an in-memory store and a fake
GitHub. No model, no Qdrant and no network are involved."""

import httpx
import pytest

from repomind.config import Settings
from repomind.core.models import RepoRef
from repomind.embeddings.errors import EmbeddingError
from repomind.ingestion.chunking import CodeChunker
from repomind.ingestion.file_filter import DefaultFileFilter
from repomind.ingestion.github_fetcher import GitHubFetcher
from repomind.ingestion.service import IngestionService
from repomind.retrieval.errors import RepositoryNotIndexed
from repomind.retrieval.indexing import IndexingService, document_text
from repomind.retrieval.search import SearchService
from tests.fakes import FakeEmbedder, InMemoryVectorStore
from tests.github_fake import SHA, FakeGitHub
from tests.sample_repo import SAMPLE_FILES

URL = "https://github.com/octo/demo"
REPO = RepoRef(owner="octo", name="demo")
OTHER_SHA = "f" * 40


@pytest.fixture
def fake_github() -> FakeGitHub:
    return FakeGitHub(files=dict(SAMPLE_FILES))


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


def make_indexer(fake_github, embedder, store) -> IndexingService:
    settings = Settings(_env_file=None)
    fetcher = GitHubFetcher(settings, DefaultFileFilter(10_000), transport=fake_github.transport())
    return IndexingService(
        IngestionService(fetcher, CodeChunker()), embedder, store, "test", batch_size=2
    )


def test_index_embeds_and_stores_every_chunk(fake_github, embedder, store) -> None:
    summary = make_indexer(fake_github, embedder, store).index(URL)

    assert summary.chunks_indexed == summary.ingestion.chunks_created == len(store.points)
    assert summary.embeddings_computed == summary.chunks_indexed
    assert summary.embeddings_reused == 0
    assert summary.embedding_dimension == 64 == store.dimension
    assert summary.embedding_model == "fake-model"
    assert {c.commit_sha for c, _, _ in store.points.values()} == {SHA}


def test_reindexing_the_same_commit_reuses_embeddings(fake_github, embedder, store) -> None:
    indexer = make_indexer(fake_github, embedder, store)
    first = indexer.index(URL)
    embedded_before = embedder.documents_embedded

    second = indexer.index(URL)

    assert second.embeddings_computed == 0
    assert second.embeddings_reused == first.chunks_indexed
    assert embedder.documents_embedded == embedded_before
    assert len(store.points) == first.chunks_indexed  # no duplicates


def test_new_commit_replaces_the_old_version(fake_github, embedder, store) -> None:
    indexer = make_indexer(fake_github, embedder, store)
    first = indexer.index(URL)

    fake_github.sha_body = OTHER_SHA
    fake_github.files["net/retry.py"] = b"def retry():\n    pass\n"
    second = indexer.index(URL)

    assert second.ingestion.commit_sha == OTHER_SHA
    assert second.stale_chunks_deleted == first.chunks_indexed
    assert {c.commit_sha for c, _, _ in store.points.values()} == {OTHER_SHA}


def test_changing_the_embedding_model_recomputes(fake_github, store) -> None:
    make_indexer(fake_github, FakeEmbedder(model_name="model-a"), store).index(URL)

    summary = make_indexer(fake_github, FakeEmbedder(model_name="model-b"), store).index(URL)

    assert summary.embeddings_reused == 0
    assert {model for _, _, model in store.points.values()} == {"model-b"}


def test_empty_repository_indexes_nothing_and_clears_old_chunks(
    fake_github, embedder, store
) -> None:
    indexer = make_indexer(fake_github, embedder, store)
    indexer.index(URL)

    fake_github.overrides["/commits/main"] = lambda r: httpx.Response(409)
    summary = indexer.index(URL)

    assert summary.chunks_indexed == 0 and summary.embeddings_computed == 0
    assert store.points == {}


def test_embedding_failure_stores_nothing(fake_github, embedder, store) -> None:
    embedder.fail = True

    with pytest.raises(EmbeddingError):
        make_indexer(fake_github, embedder, store).index(URL)
    assert store.points == {}


def test_secrets_are_not_stored_in_payloads(fake_github, embedder, store) -> None:
    make_indexer(fake_github, embedder, store).index(URL)

    texts = [c.text for c, _, _ in store.points.values()]
    assert any("[REDACTED]" in t for t in texts)
    assert not any("sk_live_" in t for t in texts)


def test_document_text_includes_path_and_symbol(fake_github, embedder, store) -> None:
    make_indexer(fake_github, embedder, store).index(URL)
    chunk = next(c for c, _, _ in store.points.values() if c.symbol == "def hash_password")

    text = document_text(chunk)

    assert text.startswith("auth/passwords.py\ndef hash_password\n\n")
    assert text.endswith(chunk.text)


# --- Search ---------------------------------------------------------------------------------


@pytest.fixture
def indexed(fake_github, embedder, store) -> SearchService:
    make_indexer(fake_github, embedder, store).index(URL)
    return SearchService(embedder, store)


# The fake embedder only matches shared words, so these queries reuse words from the code.
# Natural-language questions (tests.sample_repo.QUESTIONS) are checked with the real model in
# tests/integration/test_real_model.py.
KEYWORD_QUERIES = {
    "invoice total tax": ("billing/invoice.py", "def calculate_invoice_total"),
    "hash password salt": ("auth/passwords.py", "def hash_password"),
    "retry backoff attempts": ("net/retry.py", "def retry_with_backoff"),
}


@pytest.mark.parametrize(("question", "expected"), KEYWORD_QUERIES.items())
def test_search_finds_the_matching_function(indexed: SearchService, question, expected) -> None:
    response = indexed.search(URL, question, top_k=3)

    top = response.results[0]
    assert (top.path, top.symbol) == expected
    assert top.url.startswith(f"https://github.com/octo/demo/blob/{SHA}/{top.path}#L")


def test_search_respects_top_k(indexed: SearchService) -> None:
    assert len(indexed.search(URL, "password invoice retry", top_k=2).results) == 2


def test_search_unindexed_repository(indexed: SearchService) -> None:
    with pytest.raises(RepositoryNotIndexed):
        indexed.search("https://github.com/octo/other", "anything", top_k=3)


def test_search_unindexed_commit(indexed: SearchService) -> None:
    with pytest.raises(RepositoryNotIndexed):
        indexed.search(URL, "anything", top_k=3, commit_sha=OTHER_SHA)


def test_search_never_returns_another_repositorys_chunks(embedder, store) -> None:
    for owner_repo in ("octo/demo", "octo/other"):
        owner, name = owner_repo.split("/")
        fake = FakeGitHub(owner=owner, repo=name, files=dict(SAMPLE_FILES))
        make_indexer(fake, embedder, store).index(f"https://github.com/{owner_repo}")

    response = SearchService(embedder, store).search(URL, "password", top_k=50)

    assert response.results
    assert all(r.url.startswith("https://github.com/octo/demo/") for r in response.results)


def test_blank_query_is_rejected(indexed: SearchService) -> None:
    with pytest.raises(ValueError):
        indexed.search(URL, "   ", top_k=3)
