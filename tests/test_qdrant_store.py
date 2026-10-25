"""Tests for QdrantVectorStore.

Every test runs against Qdrant's embedded engine (`QdrantClient(":memory:")`, part of
qdrant-client). When REPOMIND_TEST_QDRANT_URL is set (e.g. http://localhost:6333), each test
also runs against that real Qdrant server; those runs are marked `integration`.
"""

import os
import uuid

import pytest
from qdrant_client import QdrantClient, models

from repomind.config import Settings
from repomind.core.models import CodeChunk, RepoRef
from repomind.vectorstore.errors import (
    CollectionMismatch,
    KeywordSearchUnavailable,
    VectorStoreUnavailable,
)
from repomind.vectorstore.qdrant import QdrantVectorStore, point_id

SERVER_URL = os.environ.get("REPOMIND_TEST_QDRANT_URL")
MODEL = "test-model"
A = RepoRef(owner="octo", name="alpha")
B = RepoRef(owner="octo", name="beta")
SHA1, SHA2 = "1" * 40, "2" * 40

backends = [pytest.param("memory", id="embedded")]
if SERVER_URL:
    backends.append(pytest.param("server", id="server", marks=pytest.mark.integration))


@pytest.fixture(params=backends)
def store(request):
    collection = f"test_{uuid.uuid4().hex[:12]}"
    if request.param == "memory":
        client = QdrantClient(":memory:")
        vector_store = QdrantVectorStore(lambda: client, collection, payload_indexes=False)
    else:
        client = QdrantClient(url=SERVER_URL, timeout=10)
        vector_store = QdrantVectorStore(lambda: client, collection)
    vector_store.ensure_collection(4)
    yield vector_store
    if request.param == "server":
        client.delete_collection(collection)
    vector_store.close()


def chunk(repo: RepoRef, path: str, sha: str = SHA1, start: int = 1, text: str = "code"):
    return CodeChunk(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_URL, f"{repo.full_name}{sha}{path}{start}").hex,
        repo=repo,
        commit_sha=sha,
        path=path,
        start_line=start,
        end_line=start + 2,
        language="python",
        symbol="def f",
        text=text,
    )


def test_upsert_and_search_returns_chunk_with_metadata(store: QdrantVectorStore) -> None:
    c = chunk(A, "src/app.py", text="def f():\n    return 1")
    store.upsert([c], [[1, 0, 0, 0]], MODEL)

    [hit] = store.search(A, [1, 0, 0, 0], limit=5, embedding_model=MODEL)

    assert hit.chunk == c
    assert hit.score == pytest.approx(1.0)


def test_results_never_include_other_repositories(store: QdrantVectorStore) -> None:
    store.upsert([chunk(A, "a.py"), chunk(B, "b.py")], [[1, 0, 0, 0], [1, 0, 0, 0]], MODEL)

    hits = store.search(A, [1, 0, 0, 0], limit=10, embedding_model=MODEL)

    assert [h.chunk.repo for h in hits] == [A]
    assert store.count(A) == 1 and store.count(B) == 1


def test_repository_identity_is_case_insensitive(store: QdrantVectorStore) -> None:
    store.upsert([chunk(A, "a.py")], [[1, 0, 0, 0]], MODEL)

    assert store.count(RepoRef(owner="OCTO", name="Alpha")) == 1


def test_reindexing_the_same_chunk_does_not_duplicate(store: QdrantVectorStore) -> None:
    c = chunk(A, "a.py")
    store.upsert([c], [[1, 0, 0, 0]], MODEL)
    store.upsert([c], [[0, 1, 0, 0]], MODEL)

    assert store.count(A) == 1


def test_existing_ids_respects_embedding_model(store: QdrantVectorStore) -> None:
    c1, c2 = chunk(A, "a.py"), chunk(A, "b.py")
    store.upsert([c1], [[1, 0, 0, 0]], MODEL)

    assert store.existing_ids([c1.chunk_id, c2.chunk_id], MODEL) == {c1.chunk_id}
    assert store.existing_ids([c1.chunk_id], "other-model") == set()
    assert store.existing_ids([], MODEL) == set()


def test_search_filters_by_embedding_model(store: QdrantVectorStore) -> None:
    store.upsert([chunk(A, "a.py")], [[1, 0, 0, 0]], "old-model")

    assert store.search(A, [1, 0, 0, 0], limit=5, embedding_model=MODEL) == []


def test_delete_other_commits_keeps_only_the_new_version(store: QdrantVectorStore) -> None:
    old = [chunk(A, "a.py", SHA1), chunk(A, "b.py", SHA1)]
    new = [chunk(A, "a.py", SHA2)]
    other_repo = [chunk(B, "a.py", SHA1)]
    store.upsert(old + new + other_repo, [[1, 0, 0, 0]] * 4, MODEL)

    deleted = store.delete_other_commits(A, SHA2)

    assert deleted == 2
    assert store.count(A) == 1 and store.count(A, commit_sha=SHA2) == 1
    assert store.count(B) == 1  # other repositories are untouched


def test_search_can_target_a_commit(store: QdrantVectorStore) -> None:
    store.upsert([chunk(A, "a.py", SHA1), chunk(A, "a.py", SHA2)], [[1, 0, 0, 0]] * 2, MODEL)

    hits = store.search(A, [1, 0, 0, 0], limit=5, embedding_model=MODEL, commit_sha=SHA2)

    assert [h.chunk.commit_sha for h in hits] == [SHA2]


def test_results_are_ordered_by_similarity(store: QdrantVectorStore) -> None:
    near, far = chunk(A, "near.py"), chunk(A, "far.py")
    store.upsert([near, far], [[1, 0.1, 0, 0], [0, 1, 0, 0]], MODEL)

    hits = store.search(A, [1, 0, 0, 0], limit=2, embedding_model=MODEL)

    assert [h.chunk.path for h in hits] == ["near.py", "far.py"]
    assert hits[0].score > hits[1].score


def test_dimension_mismatch_is_reported(store: QdrantVectorStore) -> None:
    store._ready_dimension = None

    with pytest.raises(CollectionMismatch, match="4-dimension"):
        store.ensure_collection(8)


def test_malformed_payloads_are_skipped(store: QdrantVectorStore) -> None:
    good = chunk(A, "good.py")
    store.upsert([good], [[1, 0, 0, 0]], MODEL)
    bad = models.PointStruct(
        id=str(uuid.uuid4()),
        vector=[1, 0, 0, 0],
        payload={"repo_key": "octo/alpha", "embedding_model": MODEL, "start_line": "x"},
    )
    store._get_client().upsert(store.collection, points=[bad], wait=True)

    hits = store.search(A, [1, 0, 0, 0], limit=5, embedding_model=MODEL)

    assert [h.chunk.path for h in hits] == ["good.py"]


def test_mismatched_lengths_are_rejected(store: QdrantVectorStore) -> None:
    with pytest.raises(ValueError):
        store.upsert([chunk(A, "a.py")], [], MODEL)


def test_missing_collection_means_nothing_indexed() -> None:
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(lambda: client, "never_created", payload_indexes=False)

    assert store.count(A) == 0
    assert store.search(A, [1, 0, 0, 0], 5, MODEL) == []
    assert store.delete_other_commits(A, SHA1) == 0


def test_point_ids_are_stable_uuids() -> None:
    assert point_id("0" * 31 + "1") == "00000000-0000-0000-0000-000000000001"


def test_unreachable_server_is_unavailable() -> None:
    settings = Settings(_env_file=None, qdrant_url="http://127.0.0.1:9", qdrant_timeout_seconds=1)
    store = QdrantVectorStore.from_settings(settings)

    with pytest.raises(VectorStoreUnavailable):
        store.ensure_collection(4)


def test_client_creation_failure_is_unavailable() -> None:
    def broken():
        raise RuntimeError("Storage folder is already accessed by another instance")

    with pytest.raises(VectorStoreUnavailable):
        QdrantVectorStore(broken, "c").count(A)


def test_local_storage_persists_across_restarts(tmp_path) -> None:
    settings = Settings(_env_file=None, qdrant_path=str(tmp_path / "qdrant"))
    first = QdrantVectorStore.from_settings(settings)
    first.ensure_collection(4)
    first.upsert([chunk(A, "a.py")], [[1, 0, 0, 0]], MODEL)
    first.close()

    second = QdrantVectorStore.from_settings(settings)  # a new process would do the same

    assert second.count(A) == 1
    assert second.search(A, [1, 0, 0, 0], 1, MODEL)[0].chunk.path == "a.py"
    second.close()


# --- Keyword search -------------------------------------------------------------------------


def code(repo: RepoRef, path: str, text: str, sha: str = SHA1, symbol: str | None = None):
    c = chunk(repo, path, sha, text=text)
    return c.model_copy(update={"symbol": symbol})


def index_texts(store: QdrantVectorStore, chunks: list[CodeChunk]) -> None:
    store.upsert(chunks, [[1, 0, 0, 0]] * len(chunks), MODEL)


def test_new_collections_support_keyword_search(store: QdrantVectorStore) -> None:
    assert store.supports_keyword_search()


def test_keyword_search_finds_exact_identifiers(store: QdrantVectorStore) -> None:
    target = code(
        A, "auth/passwords.py", "def hash_password(pw):\n    return pw", symbol="def hash_password"
    )
    index_texts(
        store,
        [
            code(A, "billing/invoice.py", "def calculate_invoice_total(items): ..."),
            target,
            code(A, "auth/session.py", "def password_reset(user):\n    # resets a password"),
        ],
    )

    hits = store.keyword_search(A, "hash_password", limit=5)

    assert hits[0].chunk == target
    assert hits[0].score > 0
    assert "billing/invoice.py" not in [h.chunk.path for h in hits]  # shares no term


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("GitHubFetcher", "src/fetch.py"),  # class name
        ("github_fetcher.py", "src/github_fetcher.py"),  # file name
        ("MAX_RETRY_ATTEMPTS", "src/config.py"),  # constant / variable
        ("Repository archive is too large", "src/errors.py"),  # error message
    ],
)
def test_keyword_search_kinds_of_lookup(
    store: QdrantVectorStore, query: str, expected: str
) -> None:
    index_texts(
        store,
        [
            code(A, "src/fetch.py", "class GitHubFetcher:\n    def fetch(self): ..."),
            code(A, "src/github_fetcher.py", "from fetch import download\n"),
            code(A, "src/config.py", "MAX_RETRY_ATTEMPTS = 5\nTIMEOUT = 30"),
            code(A, "src/errors.py", 'raise TooLarge("Repository archive is too large")'),
            code(A, "src/other.py", "def unrelated(): return 1"),
        ],
    )

    hits = store.keyword_search(A, query, limit=3)

    assert hits[0].chunk.path == expected


def test_keyword_search_never_returns_other_repositories(store: QdrantVectorStore) -> None:
    index_texts(
        store,
        [code(A, "a.py", "def other_thing(): ..."), code(B, "b.py", "def hash_password(): ...")],
    )

    assert store.keyword_search(A, "hash_password", limit=10) == []
    assert [h.chunk.repo for h in store.keyword_search(B, "hash_password", limit=10)] == [B]


def test_keyword_search_can_target_a_commit(store: QdrantVectorStore) -> None:
    index_texts(
        store,
        [code(A, "a.py", "def parse(): ...", SHA1), code(A, "a.py", "def parse(): ...", SHA2)],
    )

    hits = store.keyword_search(A, "parse", limit=5, commit_sha=SHA2)

    assert [h.chunk.commit_sha for h in hits] == [SHA2]
    assert len(store.keyword_search(A, "parse", limit=5)) == 2


def test_keyword_search_respects_limit_and_handles_unsearchable_queries(store) -> None:
    index_texts(store, [code(A, f"m{i}.py", "def handler(): ...") for i in range(6)])

    assert len(store.keyword_search(A, "handler", limit=4)) == 4
    assert store.keyword_search(A, "?? the is", limit=4) == []
    assert store.keyword_search(A, "nonexistentterm", limit=4) == []


def test_rare_terms_outweigh_common_ones(store: QdrantVectorStore) -> None:
    common = [code(A, f"c{i}.py", "def handler(request): return request") for i in range(5)]
    rare = code(A, "rare.py", "def handler(request): validate_signature(request)")
    index_texts(store, [*common, rare])

    hits = store.keyword_search(A, "handler validate_signature", limit=3)

    assert hits[0].chunk.path == "rare.py"


def test_existing_ids_requires_current_keyword_encoding(store: QdrantVectorStore) -> None:
    c = chunk(A, "a.py")
    index_texts(store, [c])
    assert store.existing_ids([c.chunk_id], MODEL) == {c.chunk_id}

    # Simulate a point stored by an older keyword encoding: it must be re-indexed.
    store._get_client().set_payload(
        store.collection, payload={"keyword_version": 0}, points=[point_id(c.chunk_id)], wait=True
    )

    assert store.existing_ids([c.chunk_id], MODEL) == set()
    assert store.keyword_search(A, "code", limit=5) == []  # old encoding is not searched


def test_missing_collection_keyword_search_is_empty() -> None:
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(lambda: client, "never_created", payload_indexes=False)

    assert store.supports_keyword_search()  # it will be created with keyword support
    assert store.keyword_search(A, "anything", 5) == []


def test_phase3_collection_keeps_semantic_search_but_not_keywords() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(  # exactly how Phase 3 created collections
        "legacy", vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
    )
    store = QdrantVectorStore(lambda: client, "legacy", payload_indexes=False)
    store.ensure_collection(4)
    c = chunk(A, "a.py")

    store.upsert([c], [[1, 0, 0, 0]], MODEL)  # stores without a keyword vector, no error

    assert not store.supports_keyword_search()
    assert store.existing_ids([c.chunk_id], MODEL) == {c.chunk_id}
    assert store.search(A, [1, 0, 0, 0], 5, MODEL)[0].chunk == c
    with pytest.raises(KeywordSearchUnavailable):
        store.keyword_search(A, "code", 5)


def test_chunk_without_searchable_terms_is_stored(store: QdrantVectorStore) -> None:
    c = code(A, "a/b", "= + ;")  # no term survives tokenization: an empty sparse vector

    index_texts(store, [c])

    assert store.count(A) == 1
    assert store.search(A, [1, 0, 0, 0], 5, MODEL)[0].chunk == c
