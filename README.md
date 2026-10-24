# RepoMind

RepoMind lets you point at a public GitHub repository, index its source files, and ask
natural-language questions about how it works. Every answer is grounded in retrieved source
code and cites the files and line numbers it came from.

> **Status: Phase 3 (semantic retrieval).** RepoMind can fetch a public GitHub repository,
> split it into line-numbered chunks, embed them with Sentence Transformers, store them in
> Qdrant, and return the chunks most relevant to a question. Hybrid search, reranking and
> LLM-written answers arrive in later phases.

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

The first `/index` or `/search` call downloads the embedding model (about 90 MB) from
Hugging Face and loads it, which takes a few seconds; later calls reuse the loaded model.

### Tests and linting

```bash
uv run pytest               # unit tests: no network, no model, no Qdrant server needed
uv run ruff check .
uv run ruff format --check .
```

Integration tests use the real embedding model and a real Qdrant server. They are skipped
unless you point them at both:

```bash
docker compose up -d qdrant
REPOMIND_TEST_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
REPOMIND_TEST_QDRANT_URL=http://localhost:6333 \
uv run pytest -m integration
```

### Docker

```bash
docker compose up --build   # API on port 8000 plus a Qdrant server on port 6333
```

The image runs as a non-root user, contains no `.env` file, and has a built-in health check.
Vectors persist in the `qdrant_data` volume and the downloaded model in `model_cache`.
PyTorch is the CPU-only build (see Embedding devices), which keeps the image small.

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

### Ingestion limitations

- Only the default branch of public github.com repositories. No branches, tags, sub-paths,
  private repositories, GitHub Enterprise or other hosts.
- Ingestion runs synchronously inside the request. Large repositories can take a while; a
  background job queue may come later.
- Structure-aware chunking is Python-only. Other languages use the line-based splitter.
- Git LFS files arrive as small pointer files, and submodules are not included in GitHub
  archives.
- Secret redaction uses known token formats. It reduces risk but cannot catch every secret.
- No automatic retry on GitHub errors; the API reports the error so the caller can retry.

## Indexing and searching

`POST /index` runs the ingestion above, embeds every chunk and stores it in Qdrant:

```bash
curl -X POST http://127.0.0.1:8000/index \
  -H 'Content-Type: application/json' \
  -d '{"repo_url": "https://github.com/yashG72/repomind"}'
```

```json
{
  "ingestion": {"repository": "yashG72/repomind", "commit_sha": "f410af66...", "chunks_created": 56, "...": "..."},
  "collection": "repomind_chunks",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_dimension": 384,
  "chunks_indexed": 56,
  "embeddings_computed": 56,
  "embeddings_reused": 0,
  "stale_chunks_deleted": 0
}
```

`POST /search` embeds a question and returns the closest chunks of that repository:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"repo_url": "https://github.com/yashG72/repomind", "query": "health check endpoint", "top_k": 3}'
```

```json
{
  "repository": "yashG72/repomind",
  "query": "health check endpoint",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "results": [
    {
      "score": 0.4552,
      "path": "src/repomind/api/routes/health.py",
      "start_line": 22,
      "end_line": 29,
      "symbol": "def health",
      "language": "python",
      "commit_sha": "f410af6645fa07f56f0eae3b01687beec1bca235",
      "url": "https://github.com/yashG72/repomind/blob/f410af6645fa07f56f0eae3b01687beec1bca235/src/repomind/api/routes/health.py#L22-L29",
      "text": "@router.get(\"/health\", ...)\ndef health(...): ..."
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `top_k` | Number of results, 1–50 (default `REPOMIND_SEARCH_DEFAULT_TOP_K`, 5) |
| `commit_sha` | Optional: search one specific indexed commit |
| `score` | Cosine similarity; higher is closer. Compare scores within one query, not across queries |

Search returns 404 if the repository has not been indexed, 503 if the model or Qdrant is
unavailable, and 422 for an empty or too-long query.

### How indexing works

1. **Identity.** Each repository is identified by GitHub's own spelling of `owner/name`
   (compared case-insensitively) and each indexed version by its exact commit SHA.
2. **Stable ids.** Each chunk's Qdrant point id is derived from (repository, commit, path,
   line range). Indexing the same commit twice overwrites the same points, so there are no
   duplicates.
3. **Embedding reuse.** Before embedding, RepoMind asks Qdrant which chunks of this commit are
   already stored with the current model and embeds only the missing ones. Re-indexing an
   unchanged repository computes zero embeddings.
4. **What is embedded.** The chunk's path and symbol (for example `def health`) plus its
   code, so questions that name a file or function match well.
5. **Versioning.** After the new commit is fully stored, chunks from older commits of the
   same repository are deleted, so search sees one version. Every result carries its
   `commit_sha`, so this is always visible.
6. **Collection.** One Qdrant collection (`REPOMIND_QDRANT_COLLECTION`) holds all
   repositories. It is created on first use with the model's vector size and cosine
   distance. Every query filters on the repository, so results never mix repositories.

Each stored point's payload contains `repo_key`, `repo_owner`, `repo_name`, `commit_sha`,
`path`, `start_line`, `end_line`, `language`, `symbol`, `text` and `embedding_model`. The
text has already had tokens redacted, and secret files and private keys were never read
(see the filtering rules above).

### Persistence: local storage or a Qdrant server

| Setup | Settings | Notes |
|---|---|---|
| Local storage (default) | leave `REPOMIND_QDRANT_URL` empty; data goes to `REPOMIND_QDRANT_PATH` (`data/qdrant`) | No server needed. Survives restarts. Only one process may open the folder, so run a single worker. |
| Docker Qdrant | `REPOMIND_QDRANT_URL=http://localhost:6333` | `docker compose up -d qdrant`; data in the `qdrant_data` volume |
| Qdrant Cloud | `REPOMIND_QDRANT_URL=https://<cluster>.cloud.qdrant.io` and `REPOMIND_QDRANT_API_KEY=...` | Use for multi-worker or hosted deployments |

To reset everything, stop the app and delete `data/qdrant` (local) or the collection (server).

### Choosing the embedding model

`REPOMIND_EMBEDDING_MODEL` accepts a Hugging Face model id or a path to a local model folder
(useful offline; set `REPOMIND_EMBEDDING_LOCAL_FILES_ONLY=true` to forbid downloads). The
vector size is read from the model. If you switch to a model with a different size, also set
a new `REPOMIND_QDRANT_COLLECTION`; RepoMind refuses to mix sizes and says so. Points record
which model produced them, so after switching models, re-run `/index` and search ignores
vectors from the old model.

### Embedding devices

Embeddings run on the CPU by default (`REPOMIND_EMBEDDING_DEVICE=cpu`).

- **Linux and Windows** (including the Docker image) install the CPU-only PyTorch build from
  `https://download.pytorch.org/whl/cpu`, configured in `pyproject.toml` under
  `[tool.uv.sources]`. It leaves out the several GB of CUDA libraries that PyTorch's PyPI
  wheels bundle on Linux. `cuda` is not available with this build; using a GPU would need a
  CUDA index in `pyproject.toml` and a regenerated `uv.lock`.
- **macOS** installs PyTorch from PyPI, which is CPU/MPS only; `mps` uses Apple silicon.

### Retrieval limitations

- Semantic search only; keyword (hybrid) search and reranking come in Phase 4.
- `all-MiniLM-L6-v2` reads about the first 256 tokens of each chunk and ignores the rest,
  so the end of a long chunk does not influence its vector.
- MiniLM is a general English text model, not a code-specific one; it works best for
  questions phrased like the code's names and docstrings.
- Indexing runs inside the request and on the CPU by default; large repositories take a while.
- If indexing is interrupted after the new commit is stored but before old chunks are
  deleted, both versions can briefly appear in results; re-running `/index` fixes it.

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
| Embedder | Sentence Transformers vectors for chunks and questions | 3 ✅ |
| Vector store | Qdrant storage and nearest-neighbour search | 3 ✅ |
| Hybrid search and reranker | Combine semantic and keyword matches, then reorder | 4 |
| Context builder | Pick top chunks within a token budget, label them with citations | 4 |
| LLM client | Provider-agnostic text generation | 4 |
| API | FastAPI endpoints (`/health`, `/ingest`, `/index`, `/search`) | 1–3 ✅ |

### Project layout

```
src/repomind/
  main.py                    FastAPI app factory + request logging middleware
  config.py                  Typed settings from environment variables
  logging_config.py          JSON (or text) structured logging
  api/dependencies.py        Builds components from settings (overridable in tests)
  api/errors.py              Maps errors to HTTP status codes
  api/routes/health.py       GET /health
  api/routes/ingest.py       POST /ingest
  api/routes/retrieval.py    POST /index, POST /search
  core/models.py             Shared data types (RepoRef, SourceFile, CodeChunk, ...)
  core/interfaces.py         Protocols for each pipeline component
  ingestion/github_url.py    Strict GitHub URL parser
  ingestion/github_fetcher.py  GitHub API + archive download and safe in-memory reading
  ingestion/file_filter.py   Which files are indexed, and why others are skipped
  ingestion/secrets.py       Token redaction that keeps line numbers
  ingestion/chunking.py      Python AST chunking + line-based fallback
  ingestion/service.py       Runs one ingestion and builds the summary
  ingestion/errors.py        Ingestion error types
  embeddings/sentence_transformer.py  Embedder: loads the model once, validates vectors
  vectorstore/qdrant.py      VectorStore: collection setup, upsert, scoped search, cleanup
  retrieval/indexing.py      Ingest -> embed (only new chunks) -> store -> drop old commits
  retrieval/search.py        Embed the question -> nearest chunks of one repository
tests/                       Unit tests; GitHub, the model and Qdrant are faked or embedded
tests/integration/           Real model + real Qdrant (opt-in, see above)
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
