"""Runs the evaluation questions through each retrieval mode and computes ranking metrics.

Metrics, per mode, over the questions that have relevant code:
- Recall@K: the share of a question's relevant chunks that appear in the top K results,
  averaged over questions.
- MRR (mean reciprocal rank): 1 / rank of the first relevant result (0 if none in the top
  DEPTH), averaged over questions. 1.0 means the right code is always first.
Plus, for every question, the rank of its first relevant result, and for every question the
number of results from another repository ("leaks"), which must always be 0.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from repomind.config import Settings
from repomind.core.interfaces import Embedder, Reranker, VectorStore
from repomind.ingestion.chunking import CodeChunker
from repomind.ingestion.file_filter import DefaultFileFilter
from repomind.ingestion.github_fetcher import GitHubFetcher
from repomind.ingestion.service import IngestionService
from repomind.retrieval.indexing import IndexingService
from repomind.retrieval.search import SearchConfig, SearchHit, SearchMode, SearchService
from tests.evaluation.corpus import BLOG, SHOP
from tests.evaluation.dataset import BLOG_URL, QUERIES, SHOP_URL, EvalQuery
from tests.github_fake import FakeGitHub

DEPTH = 10  # results retrieved per question
RECALL_KS = (1, 3, 5)

# name -> (mode, rerank)
MODES: dict[str, tuple[SearchMode, bool]] = {
    "semantic": (SearchMode.SEMANTIC, False),
    "hybrid (RRF)": (SearchMode.HYBRID, False),
    "hybrid + rerank": (SearchMode.HYBRID, True),
}


@dataclass
class QueryResult:
    query: EvalQuery
    first_relevant_rank: int | None
    relevant_found: dict[int, int]  # K -> relevant chunks in the top K
    leaks: int
    top: str  # "path symbol" of the first result, for reading the report


@dataclass
class ModeResult:
    name: str
    queries: list[QueryResult] = field(default_factory=list)

    @property
    def judged(self) -> list[QueryResult]:
        return [q for q in self.queries if q.query.relevant]

    def recall(self, k: int) -> float:
        judged = self.judged
        return sum(q.relevant_found[k] / len(q.query.relevant) for q in judged) / len(judged)

    @property
    def mrr(self) -> float:
        judged = self.judged
        return sum(1 / q.first_relevant_rank for q in judged if q.first_relevant_rank) / len(judged)

    @property
    def leaks(self) -> int:
        return sum(q.leaks for q in self.queries)


SYNTHETIC_REPOS: dict[str, dict[str, bytes]] = {SHOP_URL: SHOP, BLOG_URL: BLOG}


def build_index(
    embedder: Embedder,
    store: VectorStore,
    repos: dict[str, dict[str, bytes]] = SYNTHETIC_REPOS,
) -> None:
    """Index repositories (URL -> files) into the same store, via a simulated GitHub."""
    settings = Settings(_env_file=None)
    for url, files in repos.items():
        owner, name = url.removeprefix("https://github.com/").split("/")
        fake = FakeGitHub(owner=owner, repo=name, files=dict(files))
        fetcher = GitHubFetcher(settings, DefaultFileFilter(1_000_000), transport=fake.transport())
        IndexingService(IngestionService(fetcher, CodeChunker()), embedder, store, "eval").index(
            url
        )


def is_relevant(hit: SearchHit, query: EvalQuery) -> bool:
    return any(
        hit.path == path and (symbol is None or hit.symbol == symbol)
        for path, symbol in query.relevant
    )


def judge(query: EvalQuery, hits: Sequence[SearchHit]) -> QueryResult:
    repo_prefix = query.repo_url + "/"
    rank = next((i for i, h in enumerate(hits, start=1) if is_relevant(h, query)), None)
    found = {}
    for k in RECALL_KS:
        found[k] = len(
            {
                (path, symbol)
                for path, symbol in query.relevant
                for h in hits[:k]
                if h.path == path and (symbol is None or h.symbol == symbol)
            }
        )
    leaks = sum(1 for h in hits if not h.url.lower().startswith(repo_prefix.lower()))
    top = f"{hits[0].path} {hits[0].symbol or ''}".strip() if hits else "(none)"
    return QueryResult(query, rank, found, leaks, top)


def evaluate(
    embedder: Embedder,
    store: VectorStore,
    reranker: Reranker | None,
    queries: Sequence[EvalQuery] = QUERIES,
) -> list[ModeResult]:
    service = SearchService(embedder, store, reranker, SearchConfig())
    results = []
    for name, (mode, rerank) in MODES.items():
        if rerank and reranker is None:
            continue
        mode_result = ModeResult(name)
        for query in queries:
            response = service.search(
                query.repo_url, query.question, DEPTH, mode=mode, rerank=rerank
            )
            mode_result.queries.append(judge(query, response.results))
        results.append(mode_result)
    return results


def format_report(results: list[ModeResult], title: str) -> str:
    lines = [f"## {title}", ""]
    header = "| Mode | " + " | ".join(f"Recall@{k}" for k in RECALL_KS) + " | MRR | Leaks |"
    lines += [header, "|" + "---|" * (len(RECALL_KS) + 3)]
    for r in results:
        recalls = " | ".join(f"{r.recall(k):.3f}" for k in RECALL_KS)
        lines.append(f"| {r.name} | {recalls} | {r.mrr:.3f} | {r.leaks} |")
    judged = len(results[0].judged) if results else 0
    total = len(results[0].queries) if results else 0
    lines += [
        "",
        f"{judged} questions have relevant code; {total - judged} have none (no-match and "
        "isolation questions are checked for leaks only). Ranks below are of the first "
        f"relevant result in the top {DEPTH}; '-' means not found.",
        "",
        "| Question | Category | " + " | ".join(r.name for r in results) + " |",
        "|" + "---|" * (len(results) + 2),
    ]
    for i, q in enumerate(results[0].queries if results else []):
        cells = []
        for r in results:
            qr = r.queries[i]
            if q.query.relevant:
                cells.append(str(qr.first_relevant_rank or "-"))
            else:
                cells.append(f"n/a (top: {qr.top})")
        lines.append(f"| `{q.query.question}` | {q.query.category} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
