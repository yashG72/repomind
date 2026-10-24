"""Sentence Transformers implementation of the Embedder protocol.

The model is loaded lazily, on first use, and then kept for the life of the process: loading
takes seconds and hundreds of MB of memory, so it must not happen per request. A lock makes
the first load safe when several requests arrive at once.

Vectors are L2-normalized, so cosine similarity in Qdrant equals the dot product and scores
are comparable across queries.
"""

import logging
import math
import threading
from collections.abc import Sequence
from typing import Any

from repomind.embeddings.errors import EmbeddingError

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 32,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._cache_dir = cache_dir
        self._local_files_only = local_files_only
        self._model: Any = None
        self._dimension: int | None = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        self._load()
        if self._dimension is None:  # unreachable: _load sets it or raises
            raise EmbeddingError("Embedding model dimension is unknown")
        return self._dimension

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            logger.info("loading embedding model", extra={"model": self._model_name})
            try:
                # Imported here so the API starts fast and tests without the model never pay
                # for importing torch.
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(
                    self._model_name,
                    device=self._device,
                    cache_folder=self._cache_dir,
                    local_files_only=self._local_files_only,
                )
                dimension = _model_dimension(model)
            except Exception as exc:
                logger.exception(
                    "embedding model failed to load", extra={"model": self._model_name}
                )
                raise EmbeddingError(
                    f"Could not load embedding model '{self._model_name}'"
                ) from exc
            self._dimension = dimension
            self._model = model
            logger.info(
                "embedding model loaded", extra={"model": self._model_name, "dimension": dimension}
            )

    def _encode(self, texts: list[str]) -> list[list[float]]:
        self._load()
        try:
            array = self._model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            vectors = [[float(x) for x in row] for row in array]
        except Exception as exc:
            logger.exception("embedding failed", extra={"model": self._model_name})
            raise EmbeddingError("The embedding model failed to embed the text") from exc
        validate_vectors(vectors, len(texts), self.dimension)
        return vectors


def validate_vectors(vectors: list[list[float]], expected_count: int, dimension: int) -> None:
    """Reject wrong counts, wrong sizes, NaN/infinite values and all-zero vectors."""
    if len(vectors) != expected_count:
        raise EmbeddingError("The embedding model returned the wrong number of vectors")
    for vector in vectors:
        if len(vector) != dimension:
            raise EmbeddingError("The embedding model returned a vector of the wrong size")
        if not all(math.isfinite(x) for x in vector) or not any(vector):
            raise EmbeddingError("The embedding model returned an invalid vector")


def _model_dimension(model: Any) -> int:
    # Read the size from the model itself. The method was renamed in sentence-transformers 6.
    getter = getattr(model, "get_embedding_dimension", None)
    if getter is None:
        getter = model.get_sentence_embedding_dimension
    dimension = getter()
    if not dimension:  # some models only know their size after encoding something
        dimension = len(model.encode(["dimension probe"], convert_to_numpy=True)[0])
    return int(dimension)
