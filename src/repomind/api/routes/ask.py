"""POST /ask answers a question about an indexed repository, citing the code it used."""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from repomind.api.dependencies import get_answer_service
from repomind.config import Settings, get_settings
from repomind.generation.answer import AnswerService, AskResponse
from repomind.retrieval.search import SearchMode

router = APIRouter(tags=["answers"])


class AskRequest(BaseModel):
    repo_url: str = Field(max_length=300, examples=["https://github.com/yashG72/repomind"])
    question: str = Field(
        min_length=1, max_length=20_000, examples=["How are repository URLs validated?"]
    )
    # Search results to consider; default REPOMIND_ASK_DEFAULT_TOP_K. The context limits
    # (REPOMIND_ASK_MAX_CONTEXT_CHUNKS and _CHARS) decide how many of them reach the model.
    top_k: int | None = Field(default=None, ge=1, le=20)
    # The same options as POST /search.
    commit_sha: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    mode: SearchMode | None = None
    rerank: bool | None = None


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    service: Annotated[AnswerService, Depends(get_answer_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AskResponse:
    # A plain `def` endpoint: the model call blocks, so FastAPI runs it in a worker thread.
    return service.ask(
        request.repo_url,
        request.question,
        request.top_k or settings.ask_default_top_k,
        request.commit_sha,
        mode=request.mode,
        rerank=request.rerank,
    )
