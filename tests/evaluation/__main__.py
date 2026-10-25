"""Print the retrieval evaluation report.

    uv run python -m tests.evaluation          # deterministic fakes (no model downloads)
    uv run python -m tests.evaluation --real   # real embedding and reranking models

--real downloads the models on first use (about 90 MB each). Set --qdrant-url to use a Qdrant
server instead of the embedded engine.
"""

import argparse
import logging
import uuid

from qdrant_client import QdrantClient

from repomind.config import Settings
from repomind.embeddings.sentence_transformer import SentenceTransformerEmbedder
from repomind.reranking.cross_encoder import CrossEncoderReranker
from repomind.vectorstore.qdrant import QdrantVectorStore
from tests.evaluation.corpus import REPOMIND_SNAPSHOT_COMMIT, load_repomind_snapshot
from tests.evaluation.dataset import REPOMIND_QUERIES, REPOMIND_URL
from tests.evaluation.harness import build_index, evaluate, format_report
from tests.fakes import FakeEmbedder, FakeReranker


def main() -> None:
    defaults = Settings(_env_file=None)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="use the real models")
    parser.add_argument("--embedding-model", default=defaults.embedding_model)
    parser.add_argument("--rerank-model", default=defaults.rerank_model)
    parser.add_argument("--qdrant-url", default=None)
    args = parser.parse_args()
    logging.disable(logging.INFO)

    if args.real:
        embedder = SentenceTransformerEmbedder(args.embedding_model)
        reranker = CrossEncoderReranker(args.rerank_model)
        title = f"Real models: {args.embedding_model} + {args.rerank_model}"
    else:
        embedder, reranker = FakeEmbedder(), FakeReranker()
        title = "Deterministic fakes: FakeEmbedder (hashed bag of words) + FakeReranker"

    backend = "Qdrant server" if args.qdrant_url else "embedded Qdrant"
    client = QdrantClient(url=args.qdrant_url) if args.qdrant_url else QdrantClient(":memory:")
    snapshot = load_repomind_snapshot()
    datasets = [("Synthetic shop + blog", None, None)]
    if snapshot is None:
        print(f"RepoMind snapshot {REPOMIND_SNAPSHOT_COMMIT[:7]} not in this clone; skipped.\n")
    else:
        datasets.append(
            (
                f"RepoMind @ {REPOMIND_SNAPSHOT_COMMIT[:7]}",
                {REPOMIND_URL: snapshot},
                REPOMIND_QUERIES,
            )
        )
    print(f"# Retrieval evaluation: {title} ({backend})\n")
    for name, repos, queries in datasets:
        collection = f"eval_{uuid.uuid4().hex[:8]}"
        store = QdrantVectorStore(lambda: client, collection, payload_indexes=bool(args.qdrant_url))
        try:
            if repos is None:
                build_index(embedder, store)
                results = evaluate(embedder, store, reranker)
            else:
                build_index(embedder, store, repos)
                results = evaluate(embedder, store, reranker, queries)
            print(format_report(results, name))
        finally:
            if args.qdrant_url:
                client.delete_collection(collection)


if __name__ == "__main__":
    main()
