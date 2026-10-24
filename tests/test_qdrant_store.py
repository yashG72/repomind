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
from repomind.vectorstore.errors import CollectionMismatch, VectorStoreUnavailable
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
