"""Grounded answers: retrieve code, build a bounded evidence context, ask the model, and check
every citation before returning anything.

    question ─► validate ─► SearchService.search ─► ContextBuilder ─► LLMClient.generate
                                                                            │
          answer + citations ◄── resolve ids to retrieval metadata ◄── parse + validate

Retrieval is the Phase 4 SearchService, unchanged: the same repository and commit filters,
modes and reranking. The model only ever sees the excerpts and may only cite their ids
(S1, S2, ...). The application, not the model, turns ids into paths, line ranges, commits and
GitHub URLs, so a citation can never point anywhere retrieval did not. Ids that were not in
the evidence are removed and reported. An answer that cites no evidence at all is withheld
and replaced by an insufficient-evidence response.

Valid citations show which excerpts the model relied on; they do not prove that every
sentence of the answer is correct.
"""

import json
import logging
import re
import time

from pydantic import BaseModel, Field, ValidationError

from repomind.core.interfaces import LLMClient
from repomind.core.models import Citation
from repomind.generation.context import ContextBuilder, Evidence, EvidenceContext
from repomind.generation.errors import LLMNotConfigured, MalformedLLMOutput
from repomind.generation.prompts import RESPONSE_SCHEMA, SYSTEM_PROMPT, build_user_prompt
from repomind.ingestion.github_url import parse_github_repo_url
from repomind.retrieval.errors import InvalidQuery
from repomind.retrieval.search import SearchMode, SearchResponse, SearchService

logger = logging.getLogger(__name__)

NO_EVIDENCE_ANSWER = (
    "I could not find code in this repository that answers the question, so I can't give a "
    "grounded answer. Try rephrasing the question with names of files, functions or features."
)
UNCITED_ANSWER = (
    "I can't give a grounded answer: the generated answer did not cite any of the retrieved "
    "code, so it was withheld. Try a more specific question."
)
# [S1], [S1][S2] or [S1, S2]
_CITATION_GROUP = re.compile(r"\[\s*(S\d+(?:\s*,\s*S\d+)*)\s*\]")
_CITATION_ID = re.compile(r"S\d+")


class EvidenceUsed(BaseModel):
    """An excerpt that was sent to the model (its code is not repeated here)."""

    id: str
    path: str
    start_line: int
    end_line: int
    symbol: str | None
    language: str | None
    commit_sha: str
    url: str
    score: float  # the retrieval score that ranked it (see SearchHit.score)
    cited: bool


class RetrievalInfo(BaseModel):
    mode: SearchMode
    embedding_model: str
    reranker_model: str | None
    results_retrieved: int  # search results considered
    evidence_used: int  # excerpts sent to the model
    duplicates_dropped: int
    dropped_over_budget: int  # left out whole because the context limit was reached
    context_chars: int
    search_duration_ms: float


class AskResponse(BaseModel):
    repository: str
    question: str
    answer: str
    # True when the answer says (or the application decided) that the retrieved code is
    # not enough to answer the question.
    insufficient_evidence: bool
    citations: list[Citation]
    evidence: list[EvidenceUsed]
    retrieval: RetrievalInfo
    llm_model: str | None  # null when no model was called (nothing to answer from)
    warnings: list[str] = Field(default_factory=list)
    duration_ms: float


class _ModelAnswer(BaseModel):
    """The JSON the model must return (see prompts.RESPONSE_SCHEMA)."""

    answer: str
    citations: list[str]
    insufficient_evidence: bool


class AnswerService:
    def __init__(
        self,
        search: SearchService,
        llm: LLMClient | None,
        context_builder: ContextBuilder,
        max_question_chars: int = 2_000,
    ) -> None:
        self._search = search
        self._llm = llm
        self._context_builder = context_builder
        self._max_question_chars = max_question_chars

    def ask(
        self,
        repo_url: str,
        question: str,
        top_k: int,
        commit_sha: str | None = None,
        mode: SearchMode | None = None,
        rerank: bool | None = None,
    ) -> AskResponse:
        """Raises InvalidRepositoryURL, InvalidQuery, LLMNotConfigured, RepositoryNotIndexed,
        the search dependencies' errors, and LLMError subclasses when the model fails."""
        started = time.perf_counter()
        parse_github_repo_url(repo_url)  # reject a bad URL before any other work
        question = question.strip()
        if not question:
            raise InvalidQuery("The question must not be empty")
        if len(question) > self._max_question_chars:
            raise InvalidQuery(f"The question is longer than {self._max_question_chars} characters")
        if self._llm is None:
            raise LLMNotConfigured(
                "Answer generation is not configured: set REPOMIND_LLM_PROVIDER and "
                "REPOMIND_LLM_API_KEY"
            )

        found = self._search.search(repo_url, question, top_k, commit_sha, mode=mode, rerank=rerank)
        context = self._context_builder.build(found.repository, found.results)
        warnings = list(found.warnings)
        if context.dropped_over_budget:
            warnings.append(
                f"{context.dropped_over_budget} retrieved excerpt(s) did not fit in the "
                "context limit and were left out"
            )

        if not context.evidence:
            warnings.append("No retrieved code could be used as evidence for this question.")
            return self._respond(
                found, question, NO_EVIDENCE_ANSWER, True, [], context, None, warnings, started
            )

        raw = self._llm.generate(
            SYSTEM_PROMPT, build_user_prompt(question, context.evidence), RESPONSE_SCHEMA
        )
        parsed = _parse(raw)
        answer, cited, unknown = _resolve_citations(parsed, context.by_id())
        if unknown:
            warnings.append(
                "The model cited sources that were not in the evidence; they were removed: "
                + ", ".join(unknown)
            )
        insufficient = parsed.insufficient_evidence
        if not cited and not insufficient:
            # An answer with no supporting evidence is not returned as if it were grounded.
            answer, insufficient = UNCITED_ANSWER, True
            warnings.append("The generated answer cited no retrieved evidence and was withheld.")
        return self._respond(
            found,
            question,
            answer,
            insufficient,
            cited,
            context,
            self._llm.model_name,
            warnings,
            started,
        )

    def _respond(
        self,
        found: SearchResponse,
        question: str,
        answer: str,
        insufficient: bool,
        cited: list[Evidence],
        context: EvidenceContext,
        llm_model: str | None,
        warnings: list[str],
        started: float,
    ) -> AskResponse:
        cited_ids = {e.id for e in cited}
        duration = round((time.perf_counter() - started) * 1000, 2)
        # The question, prompt and code are never logged; only sizes and counts.
        logger.info(
            "question answered",
            extra={
                "repo": found.repository,
                "question_chars": len(question),
                "evidence": len(context.evidence),
                "context_chars": context.chars,
                "citations": len(cited),
                "insufficient_evidence": insufficient,
                "warnings": len(warnings),
                "duration_ms": duration,
            },
        )
        return AskResponse(
            repository=found.repository,
            question=question,
            answer=answer,
            insufficient_evidence=insufficient,
            citations=[
                Citation(
                    id=e.id,
                    repository=e.repository,
                    path=e.path,
                    start_line=e.start_line,
                    end_line=e.end_line,
                    commit_sha=e.commit_sha,
                    url=e.url,
                    symbol=e.symbol,
                )
                for e in cited
            ],
            evidence=[
                EvidenceUsed(
                    id=e.id,
                    path=e.path,
                    start_line=e.start_line,
                    end_line=e.end_line,
                    symbol=e.symbol,
                    language=e.language,
                    commit_sha=e.commit_sha,
                    url=e.url,
                    score=e.score,
                    cited=e.id in cited_ids,
                )
                for e in context.evidence
            ],
            retrieval=RetrievalInfo(
                mode=found.mode,
                embedding_model=found.embedding_model,
                reranker_model=found.reranker_model,
                results_retrieved=len(found.results),
                evidence_used=len(context.evidence),
                duplicates_dropped=context.duplicates_dropped,
                dropped_over_budget=context.dropped_over_budget,
                context_chars=context.chars,
                search_duration_ms=found.duration_ms,
            ),
            llm_model=llm_model,
            warnings=warnings,
            duration_ms=duration,
        )


def _parse(raw: str) -> _ModelAnswer:
    """Parse the model's JSON. Tolerates a Markdown code fence around it, nothing else."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    try:
        parsed = _ModelAnswer.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise MalformedLLMOutput("The model's reply was not in the expected format") from exc
    if not parsed.answer.strip():
        raise MalformedLLMOutput("The model returned an empty answer")
    return parsed


def _resolve_citations(
    parsed: _ModelAnswer, evidence: dict[str, Evidence]
) -> tuple[str, list[Evidence], list[str]]:
    """Return the answer with unknown citation markers removed, the cited evidence in order of
    first mention, and the unknown ids."""
    order: list[str] = []
    unknown: list[str] = []

    def note(citation_id: str) -> bool:
        known = citation_id in evidence
        target = order if known else unknown
        if citation_id not in target:
            target.append(citation_id)
        return known

    def rewrite(match: re.Match[str]) -> str:
        kept = [i for i in _CITATION_ID.findall(match.group(1)) if note(i)]
        return "".join(f"[{i}]" for i in kept)

    answer = _CITATION_GROUP.sub(rewrite, parsed.answer).strip()
    for citation_id in parsed.citations:
        note(citation_id.strip().strip("[]").strip())
    return answer, [evidence[i] for i in order], unknown
