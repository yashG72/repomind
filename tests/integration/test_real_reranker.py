"""The real cross-encoder reranker, alone and in the full hybrid pipeline.

Skipped unless REPOMIND_TEST_RERANK_MODEL is set to a model id or local model folder:

    REPOMIND_TEST_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2 \\
    REPOMIND_TEST_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \\
    REPOMIND_TEST_QDRANT_URL=http://localhost:6333 \\
    uv run pytest -m integration

The pipeline test also needs REPOMIND_TEST_EMBEDDING_MODEL. Without REPOMIND_TEST_QDRANT_URL it
uses Qdrant's embedded engine. GitHub is simulated, so the expected answers are known.
"""

import os
import uuid

import pytest
from qdrant_client import QdrantClient

from repomind.config import Settings
from repomind.core.models import CodeChunk, RepoRef, ScoredChunk
from repomind.embeddings.sentence_transformer import SentenceTransformerEmbedder
from repomind.ingestion.chunking import CodeChunker
from repomind.ingestion.file_filter import DefaultFileFilter
from repomind.ingestion.github_fetcher import GitHubFetcher
from repomind.ingestion.service import IngestionService
from repomind.reranking.cross_encoder import CrossEncoderReranker
from repomind.reranking.errors import RerankerError
from repomind.retrieval.indexing import IndexingService
from repomind.retrieval.search import SearchService
from repomind.vectorstore.qdrant import QdrantVectorStore
from tests.github_fake import FakeGitHub
from tests.sample_repo import QUESTIONS, SAMPLE_FILES

RERANK_MODEL = os.environ.get("REPOMIND_TEST_RERANK_MODEL")
EMBEDDING_MODEL = os.environ.get("REPOMIND_TEST_EMBEDDING_MODEL")
QDRANT_URL = os.environ.get("REPOMIND_TEST_QDRANT_URL")
REPO = RepoRef(owner="octo", name="demo")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not RERANK_MODEL, reason="set REPOMIND_TEST_RERANK_MODEL to run"),
]


@pytest.fixture(scope="module")
def reranker() -> CrossEncoderReranker:
    assert RERANK_MODEL
    return CrossEncoderReranker(RERANK_MODEL)


def candidate(path: str, text: str) -> ScoredChunk:
    chunk = CodeChunk(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_URL, path).hex,
        repo=REPO,
        commit_sha="a" * 40,
        path=path,
        start_line=1,
        end_line=text.count("\n") + 1,
        text=text,
    )
    return ScoredChunk(chunk=chunk, score=0.0)


def test_real_model_puts_the_relevant_snippet_first(reranker) -> None:
    files = {path: data.decode() for path, data in SAMPLE_FILES.items() if path.endswith(".py")}
    candidates = [candidate(path, text) for path, text in sorted(files.items())]

    for question, (path, _) in QUESTIONS.items():
        ranked = reranker.rerank(question, candidates)
        assert len(ranked) == len(candidates)
        assert ranked[0].chunk.path == path, question
        assert ranked[0].score > ranked[-1].score


def test_missing_model_is_a_clear_error() -> None:
    reranker = CrossEncoderReranker(f"/nonexistent/{uuid.uuid4().hex}", local_files_only=True)

    with pytest.raises(RerankerError, match="Could not load reranking model"):
        reranker.rerank("q", [candidate("a.py", "x = 1")])


@pytest.mark.skipif(not EMBEDDING_MODEL, reason="set REPOMIND_TEST_EMBEDDING_MODEL to run")
def test_hybrid_pipeline_with_real_models(tmp_path, reranker) -> None:
    assert EMBEDDING_MODEL
    settings = Settings(
        _env_file=None,
        qdrant_url=QDRANT_URL,
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection=f"it_{uuid.uuid4().hex[:12]}",
    )
    embedder = SentenceTransformerEmbedder(EMBEDDING_MODEL)
    store = QdrantVectorStore.from_settings(settings)
    try:
        for name in ("demo", "other"):  # same code in two repositories
            fake = FakeGitHub(owner="octo", repo=name, files=dict(SAMPLE_FILES))
            fetcher = GitHubFetcher(settings, DefaultFileFilter(10_000), transport=fake.transport())
            IndexingService(IngestionService(fetcher, CodeChunker()), embedder, store, "it").index(
                f"https://github.com/octo/{name}"
            )

        search = SearchService(embedder, store, reranker)
        for question, (path, symbol) in QUESTIONS.items():
            response = search.search("https://github.com/octo/demo", question, top_k=3, rerank=True)
            assert response.reranker_model == RERANK_MODEL
            assert all(h.url.startswith("https://github.com/octo/demo/") for h in response.results)
            top = response.results[0]
            assert (top.path, top.symbol) == (path, symbol), question
            assert top.scores.rerank == pytest.approx(top.score, abs=1e-5)
    finally:
        if QDRANT_URL:
            QdrantClient(url=QDRANT_URL).delete_collection(settings.qdrant_collection)
        store.close()
