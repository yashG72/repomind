"""POST /ingest: fetch, filter and chunk a public GitHub repository without storing it.

Useful to preview what indexing would do. POST /index also stores the chunks.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from repomind.api.dependencies import get_ingestion_service
from repomind.ingestion.service import IngestionService, IngestionSummary

router = APIRouter(tags=["ingestion"])


class IngestRequest(BaseModel):
    repo_url: str = Field(max_length=300, examples=["https://github.com/yashG72/repomind"])


@router.post("/ingest", response_model=IngestionSummary)
def ingest(
    request: IngestRequest,
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
) -> IngestionSummary:
    # A plain `def` endpoint: FastAPI runs it in a worker thread, so the blocking download
    # does not stall other requests.
    return service.ingest(request.repo_url).summary
