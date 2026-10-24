# RepoMind

RepoMind lets you point at a public GitHub repository, index its source files, and ask
natural-language questions about how it works. Every answer is grounded in retrieved source
code and cites the files and line numbers it came from.

> **Status: Phase 2 (ingestion).** RepoMind can fetch a public GitHub repository, filter its
> files safely and split them into line-numbered chunks. Embeddings, vector search and
> question answering arrive in later phases.

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
uv run pytest               # no network needed: GitHub is simulated
uv run ruff check .
uv run ruff format --check .
```

### Docker

```bash
docker compose up --build   # builds the image and serves the API on port 8000
```

The image runs as a non-root user, contains no `.env` file, and has a built-in health check.

## Ingesting a repository

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"repo_url": "https://github.com/yashG72/repomind"}'
```

Example response (abridged, from a real run against this repository):

```json
{
  "repository": "yashG72/repomind",
  "commit_sha": "f410af6645fa07f56f0eae3b01687beec1bca235",
  "default_branch": "main",
  "files_discovered": 28,
  "files_processed": 17,
  "files_skipped": 11,
  "chunks_created": 56,
  "skipped_by_reason": {"empty": 5, "lock_file": 1, "unsupported_type": 5},
  "languages": {"python": 13, "dockerfile": 1, "markdown": 1, "yaml": 1, "toml": 1},
  "secrets_redacted": 0,
  "bytes_downloaded": 60486,
  "skipped_files": [{"path": "uv.lock", "reason": "lock_file"}],
  "skipped_files_truncated": false
}
```

`skipped_files` lists up to 100 skipped files with their reason. In Phase 2 the chunks are
created and counted but not stored; Phase 3 stores them in Qdrant.

| Status | Meaning |
|---|---|
| 200 | Ingested (an empty repository returns a summary with zero files) |
| 404 | Repository does not exist or is not public |
| 413 | Repository is over a size limit |
| 422 | URL is not `https://github.com/<owner>/<repo>`, or the request body is invalid |
| 429 | GitHub rate limit reached; `Retry-After` says when to try again |
| 502 | GitHub returned something unreadable, or redirected somewhere unsafe |
| 503 | GitHub is unreachable, timed out, or returned a server error |

### How ingestion works

1. **Validate the URL** (`ingestion/github_url.py`) into an owner and repository name.
2. **Fetch** (`ingestion/github_fetcher.py`) with three GitHub API requests: repository
   metadata (must be public and under the size limit), the default branch's commit SHA, and a
   `tar.gz` archive of that commit. One archive download costs the same 3 requests however
   many files the repository has, which matters because unauthenticated clients get 60
   requests per hour.
3. **Read the archive in memory.** Nothing is written to disk or executed. Only regular files
   are read; symlinks, hard links and devices are skipped.
4. **Filter** (`ingestion/file_filter.py`) by path, size and content, recording a reason for
   every skipped file.
5. **Redact** (`ingestion/secrets.py`) credential-like tokens in kept files, on the same line,
   so line numbers do not change.
6. **Chunk** (`ingestion/chunking.py`). Python files are split by structure with the standard
   `ast` module (one chunk per top-level function or class, large classes per method).
   Everything else is split by lines, preferring blank-line boundaries. Chunks never overlap
   and their line numbers match the file on GitHub.

### File filtering rules

| Rule | Examples | Reason code |
|---|---|---|
| Path escapes the repository | `../x`, `/abs`, backslashes, control characters | `unsafe_path` |
| Not a regular file | symlinks, hard links | `not_regular_file` |
| Dependency, build or cache folder | `node_modules/`, `vendor/`, `.venv/`, `dist/`, `build/`, `target/`, `__pycache__/` | `excluded_directory` |
| Likely secret | `.env`, `.env.*` (not `.env.example`), `*.pem`, `*.key`, `id_rsa`, `credentials.json`, files containing a private key | `secret_file` |
| Lock file | `package-lock.json`, `poetry.lock`, `uv.lock`, `go.sum` | `lock_file` |
| Generated | `*.min.js`, `*.map`, `*_pb2.py`, `*.pb.go`, files marked "Code generated ... DO NOT EDIT" or `@generated` | `generated` |
| Minified | any line over 5,000 characters, or average line over 300 | `minified` |
| Not a supported source or text type | images, archives, executables, fonts | `unsupported_type` |
| Binary or not UTF-8 | NUL bytes, invalid UTF-8 | `binary`, `not_utf8` |
| Over the per-file limit, or empty | | `too_large`, `empty` |
| Over the file-count limit | | `file_limit` |

The supported languages and full lists are in `src/repomind/ingestion/file_filter.py`.

### Limitations

- Only the default branch of public github.com repositories. No branches, tags, sub-paths,
  private repositories, GitHub Enterprise or other hosts.
- Ingestion runs synchronously inside the request. Large repositories can take a while; a
  background job queue may come later.
- Structure-aware chunking is Python-only. Other languages use the line-based splitter.
- Git LFS files arrive as small pointer files, and submodules are not included in GitHub
  archives.
- Secret redaction uses known token formats. It reduces risk but cannot catch every secret.
- No automatic retry on GitHub errors; the API reports the error so the caller can retry.

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
| Repository fetcher | Download files from GitHub, enforce size limits, no code execution | 2 ✅ |
| File filter | Skip binaries, secrets, generated and vendored files | 2 ✅ |
| Chunker | Split files into chunks that keep path and line numbers | 2 ✅ |
| Embedder | Sentence Transformers vectors for chunks and questions | 3 |
| Vector store | Qdrant storage and nearest-neighbour search | 3 |
| Hybrid search and reranker | Combine semantic and keyword matches, then reorder | 4 |
| Context builder | Pick top chunks within a token budget, label them with citations | 4 |
| LLM client | Provider-agnostic text generation | 4 |
| API | FastAPI endpoints (`/health`, `/ingest`) | 1–2 ✅ |

### Project layout

```
src/repomind/
  main.py                    FastAPI app factory + request logging middleware
  config.py                  Typed settings from environment variables
  logging_config.py          JSON (or text) structured logging
  api/dependencies.py        Builds components from settings (overridable in tests)
  api/routes/health.py       GET /health
  api/routes/ingest.py       POST /ingest
  core/models.py             Shared data types (RepoRef, SourceFile, CodeChunk, ...)
  core/interfaces.py         Protocols for each pipeline component
  ingestion/github_url.py    Strict GitHub URL parser
  ingestion/github_fetcher.py  GitHub API + archive download and safe in-memory reading
  ingestion/file_filter.py   Which files are indexed, and why others are skipped
  ingestion/secrets.py       Token redaction that keeps line numbers
  ingestion/chunking.py      Python AST chunking + line-based fallback
  ingestion/service.py       Runs one ingestion and builds the summary
  ingestion/errors.py        Ingestion error types
tests/                       Pytest suite; GitHub is simulated (tests/github_fake.py)
```

## Security model

- **Only public GitHub repositories.** URLs must match `https://github.com/<owner>/<repo>`.
  Ports, credentials, query strings, other hosts, IP addresses and look-alike domains are
  rejected.
- **No server-side request forgery.** RepoMind never requests the URL a user typed. It
  extracts the owner and repository name and builds GitHub API URLs itself. Redirects are
  followed manually, at most 3 times, and only to `https` URLs on `api.github.com` or
  `codeload.github.com`; anything else is refused.
- **Repository content is untrusted.** Archives are read in memory and never extracted to
  disk, so a hostile archive cannot overwrite files. Paths with `..`, absolute paths and
  links are skipped. Python files are parsed with `ast`, which reads code without running it.
  Later phases also treat content as untrusted when placing it in LLM prompts.
- **Limits.** Per-file size, total uncompressed size, compressed download size, archive entry
  count and file count are all capped. The repository size reported by GitHub is checked
  before downloading.
- **Secrets.** No keys in code; `.env` is excluded from git and Docker images. The optional
  GitHub token is sent only to `api.github.com`, never logged, and never included in error
  messages. Secret-looking files are skipped and tokens inside source files are redacted.
