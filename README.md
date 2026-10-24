# RepoMind

RepoMind lets you point at a public GitHub repository, index its source files, and ask
natural-language questions about how it works. Every answer is grounded in retrieved source
code and cites the files and line numbers it came from.

> **Status: Phase 1 (foundation).** The API skeleton, configuration, logging, health check,
> GitHub URL validation and Docker image are in place. Fetching, indexing and question
> answering arrive in later phases.

## Quick start

Requirements: Python 3.12+ and [uv](https://docs.astral.sh/uv/). Docker is optional.

```bash
uv sync                     # creates .venv and installs locked dependencies
cp .env.example .env        # optional: every setting has a safe default
uv run uvicorn repomind.main:app --reload
curl http://127.0.0.1:8000/health
# {"status":"ok","app":"RepoMind","version":"0.1.0","environment":"development"}
```

Interactive API docs are served at http://127.0.0.1:8000/docs.

### Tests and linting

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

### Docker

```bash
docker compose up --build   # builds the image and serves the API on port 8000
```

The image runs as a non-root user, contains no `.env` file, and has a built-in health check.

## Configuration

All settings are environment variables prefixed with `REPOMIND_`, loaded by
`src/repomind/config.py`. See [`.env.example`](.env.example) for the full list with defaults.
API keys are stored as `SecretStr`, so they are masked if settings are ever printed or logged.
`.env` is git-ignored and docker-ignored; never commit real secrets.

## Architecture

RepoMind is a plain Python pipeline with one component per responsibility. Components talk
through small typed interfaces (`src/repomind/core/interfaces.py`), so each can be tested on
its own and swapped (for example, a different LLM provider) without touching the others.

```
                    INDEXING (once per repository)
 GitHub URL ─► URL validation ─► Repository fetcher ─► File filter ─► Chunker
                                                                        │
                                                         Embedder ◄─────┘
                                                            │
                                                            ▼
                                                     Vector store (Qdrant)
                                                            ▲
                    QUESTION ANSWERING (per question)       │
 Question ─► Embedder ─► Hybrid search (vector + keyword) ──┘
                              │
                              ▼
                          Reranker ─► Context builder ─► LLM client ─► Answer + citations
```

| Component | Responsibility | Phase |
|---|---|---|
| URL validation | Accept only `https://github.com/<owner>/<repo>` | 1 ✅ |
| Repository fetcher | Download files from GitHub, enforce size limits, no code execution | 2 |
| File filter | Skip binaries, secrets, generated and vendored files | 2 |
| Chunker | Split files into chunks that keep path and line numbers | 2 |
| Embedder | Sentence Transformers vectors for chunks and questions | 3 |
| Vector store | Qdrant storage and nearest-neighbour search | 3 |
| Hybrid search and reranker | Combine semantic and keyword matches, then reorder | 4 |
| Context builder | Pick top chunks within a token budget, label them with citations | 4 |
| LLM client | Provider-agnostic text generation | 4 |
| API | FastAPI endpoints (`/health` today) | 1 ✅ |

### Project layout

```
src/repomind/
  main.py              FastAPI app factory + request logging middleware
  config.py            Typed settings from environment variables
  logging_config.py    JSON (or text) structured logging
  api/routes/health.py GET /health
  core/models.py       Shared data types (RepoRef, SourceFile, CodeChunk, Citation, ...)
  core/interfaces.py   Protocols for each pipeline component
  ingestion/github_url.py  Strict GitHub URL parser
tests/                 Pytest suite
```

## Security model

- **Only public GitHub repositories.** URLs must match `https://github.com/<owner>/<repo>`.
  Ports, credentials, query strings, other hosts, IP addresses and look-alike domains are
  rejected.
- **No server-side request forgery.** RepoMind never requests the URL a user typed. It
  extracts the owner and repository name, then (from Phase 2) builds GitHub requests itself
  against a fixed host allowlist, refusing redirects to any other host.
- **Repository content is untrusted.** Code is read as text only and never executed. Later
  phases also treat it as untrusted when placing it in LLM prompts.
- **Limits.** File size, repository size and file count limits are configurable.
- **Secrets.** No keys in code; `.env` is excluded from git and Docker images.
