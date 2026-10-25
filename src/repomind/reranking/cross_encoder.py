"""Cross-encoder implementation of the Reranker protocol.

Why rerank: the embedder turns the question and each chunk into vectors *separately*, which
is fast enough to search thousands of chunks but loses detail. A cross-encoder reads the
question and one chunk *together* and outputs a single relevance score. That is more
accurate, but far slower, so it only reorders a short list of candidates that semantic and
keyword search already found.

The default model, `cross-encoder/ms-marco-MiniLM-L6-v2`, is small (about 23M parameters,
about 90 MB) and runs on a CPU in tens of milliseconds per candidate. It was trained on web
search questions and passages (MS MARCO), not on code, so it understands English questions
and the comments, docstrings and names in code better than the code's logic. The README
discusses alternatives.

Like the embedder, the model is loaded lazily on first use, under a lock, and then reused for
the life of the process. Models that need `trust_remote_code` are refused, because that would
run Python code downloaded with the model. Retrieved code is only ever passed to the model as
text; it is never executed.
"""

import logging
import math
import threading
from collections.abc import Sequence
from typing import Any

from repomind.core.models import ScoredChunk, document_text
from repomind.reranking.errors import RerankerError

logger = logging.getLogger(__name__)

# The model reads at most max_length tokens of each (question, chunk) pair anyway, so very
# long chunks are cut first; this bounds tokenization time and memory per request.
MAX_DOCUMENT_CHARS = 4_000


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_candidates: int = 20,
        batch_size: int = 16,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._max_candidates = max_candidates
        self._batch_size = batch_size
        self._cache_dir = cache_dir
        self._local_files_only = local_files_only
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        if len(candidates) > self._max_candidates:
            raise ValueError(
                f"at most {self._max_candidates} candidates can be reranked, got {len(candidates)}"
            )
        if not candidates:
            return []
        self._load()
        pairs = [(query, document_text(c.chunk)[:MAX_DOCUMENT_CHARS]) for c in candidates]
        try:
            raw = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            scores = [float(x) for x in raw]
        except Exception as exc:
            logger.exception("reranking failed", extra={"model": self._model_name})
            raise RerankerError("The reranking model failed to score the results") from exc
        return order_by_scores(candidates, scores)

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            logger.info("loading reranking model", extra={"model": self._model_name})
            try:
                # Imported here so the API starts fast and tests never pay for importing torch.
                from sentence_transformers import CrossEncoder

                model = CrossEncoder(
                    self._model_name,
                    device=self._device,
                    cache_folder=self._cache_dir,
                    local_files_only=self._local_files_only,
                    trust_remote_code=False,
                )
            except Exception as exc:
                logger.exception(
                    "reranking model failed to load", extra={"model": self._model_name}
                )
                raise RerankerError(f"Could not load reranking model '{self._model_name}'") from exc
            self._model = model
            logger.info("reranking model loaded", extra={"model": self._model_name})


def order_by_scores(
    candidates: Sequence[ScoredChunk], scores: Sequence[float]
) -> list[ScoredChunk]:
    """Pair each candidate with its reranker score and sort best first.

    Rejects a wrong number of scores and NaN/infinite scores, so a broken model can never
    silently drop, duplicate or misorder results. Equal scores keep the incoming order.
    """
    if len(scores) != len(candidates):
        raise RerankerError("The reranking model returned the wrong number of scores")
    if not all(isinstance(s, int | float) and math.isfinite(s) for s in scores):
        raise RerankerError("The reranking model returned an invalid score")
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])
    return [ScoredChunk(chunk=candidates[i].chunk, score=float(scores[i])) for i in order]
