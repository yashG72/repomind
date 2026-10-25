"""Builds pipeline components for each request. Tests replace these with FastAPI's
dependency_overrides, for example to inject a fake GitHub or a fake embedder.

The embedder, reranker, vector store and LLM client are expensive (loaded models, an open
database, an HTTP connection pool), so they are created once per app in `create_app` and
shared via `app.state`. Everything else is cheap and created per request, so no repository
data outlives a request.
"""

from typing import Annotated

from fastapi import Depends, Request

from repomind.config import Settings, get_settings
from repomind.core.interfaces import Embedder, LLMClient, Reranker, VectorStore
from repomind.generation.answer import AnswerService
from repomind.generation.context import ContextBuilder
from repomind.ingestion.chunking import CodeChunker
from repomind.ingestion.file_filter import DefaultFileFilter
from repomind.ingestion.github_fetcher import GitHubFetcher
from repomind.ingestion.service import IngestionService
from repomind.retrieval.indexing import IndexingService
from repomind.retrieval.search import SearchConfig, SearchService

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_reranker(request: Request) -> Reranker:
    return request.app.state.reranker


def get_llm_client(request: Request) -> LLMClient | None:
    """None when no LLM provider is configured; POST /ask then answers 503."""
    return request.app.state.llm_client


def get_ingestion_service(settings: SettingsDep) -> IngestionService:
    fetcher = GitHubFetcher(settings, DefaultFileFilter(settings.max_file_size_bytes))
    chunker = CodeChunker(settings.chunk_max_lines, settings.chunk_max_chars)
    return IngestionService(fetcher, chunker)


def get_indexing_service(
    settings: SettingsDep,
    ingestion: Annotated[IngestionService, Depends(get_ingestion_service)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    store: Annotated[VectorStore, Depends(get_vector_store)],
) -> IndexingService:
    return IndexingService(
        ingestion, embedder, store, settings.qdrant_collection, settings.upsert_batch_size
    )


def get_search_service(
    settings: SettingsDep,
    embedder: Annotated[Embedder, Depends(get_embedder)],
    store: Annotated[VectorStore, Depends(get_vector_store)],
    reranker: Annotated[Reranker, Depends(get_reranker)],
) -> SearchService:
    return SearchService(embedder, store, reranker, SearchConfig.from_settings(settings))


def get_answer_service(
    settings: SettingsDep,
    search: Annotated[SearchService, Depends(get_search_service)],
    llm: Annotated[LLMClient | None, Depends(get_llm_client)],
) -> AnswerService:
    return AnswerService(
        search,
        llm,
        ContextBuilder(settings.ask_max_context_chunks, settings.ask_max_context_chars),
        max_question_chars=settings.ask_max_question_chars,
    )
