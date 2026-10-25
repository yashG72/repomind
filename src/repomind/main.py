"""FastAPI application factory and request logging middleware."""

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from repomind import __version__
from repomind.api.errors import register_exception_handlers
from repomind.api.routes import ask, health, ingest, retrieval
from repomind.config import Settings, get_settings
from repomind.core.interfaces import LLMClient
from repomind.embeddings.sentence_transformer import SentenceTransformerEmbedder
from repomind.generation.anthropic_client import AnthropicLLMClient
from repomind.logging_config import configure_logging
from repomind.reranking.cross_encoder import CrossEncoderReranker
from repomind.vectorstore.qdrant import QdrantVectorStore

logger = logging.getLogger("repomind.request")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)

    # Shared, long-lived resources. All are lazy: the models load and the database opens on
    # first use, so starting the app (and running tests) stays fast.
    vector_store = QdrantVectorStore.from_settings(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        vector_store.close()

    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.dependency_overrides[get_settings] = lambda: settings
    app.state.embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        cache_dir=settings.embedding_cache_dir,
        local_files_only=settings.embedding_local_files_only,
    )
    app.state.reranker = CrossEncoderReranker(
        settings.rerank_model,
        device=settings.embedding_device,
        max_candidates=settings.rerank_max_candidates,
        cache_dir=settings.embedding_cache_dir,
        local_files_only=settings.embedding_local_files_only,
    )
    app.state.vector_store = vector_store
    app.state.llm_client = _llm_client(settings)
    register_exception_handlers(app)

    @app.middleware("http")
    async def log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                },
            )
            raise
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["x-request-id"] = request_id
        logger.info(
            "request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response

    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(retrieval.router)
    app.include_router(ask.router)
    return app


def _llm_client(settings: Settings) -> LLMClient | None:
    """The configured answer provider, or None (POST /ask then answers 503). Creating it makes
    no network call; the SDK client is created on the first question."""
    if settings.llm_provider == "none":
        return None
    if settings.llm_api_key is None:
        logger.warning("REPOMIND_LLM_PROVIDER is set but REPOMIND_LLM_API_KEY is not; /ask is off")
        return None
    return AnthropicLLMClient.from_settings(settings)


app = create_app()
