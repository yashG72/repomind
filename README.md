# RepoMind

RepoMind lets you point at a public GitHub repository, index its source files, and ask
natural-language questions about how it works. Every answer is grounded in retrieved source
code and cites the files and line numbers it came from.

> **Status: Phase 4 (hybrid retrieval).** RepoMind can fetch a public GitHub repository,
> split it into line-numbered chunks, embed them with Sentence Transformers, store them in
> Qdrant, and return the chunks most relevant to a question. Search combines semantic
> (meaning-based) and keyword (exact-name) retrieval, merges them with Reciprocal Rank Fusion
> and reranks the best candidates with a local cross-encoder model. LLM-written answers arrive
> in a later phase.

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
Hugging Face and loads it, which takes a few seconds; later calls reuse the loaded model. The
first hybrid search also downloads and loads the reranking model (about 90 MB).

### Tests and linting

```bash
uv run pytest               # unit tests: no network, no model, no Qdrant server needed
uv run ruff check .
uv run ruff format --check .
```

Integration tests use the real models and a real Qdrant server. Each part is skipped unless
you point it at the real thing:

```bash
docker compose up -d qdrant
REPOMIND_TEST_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
REPOMIND_TEST_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2 \
REPOMIND_TEST_QDRANT_URL=http://localhost:6333 \
uv run pytest -m integration
```

The retrieval evaluation (see [Evaluation](#evaluation)) prints a report comparing the search
modes:

```bash
uv run python -m tests.evaluation          # deterministic fakes, no downloads
uv run python -m tests.evaluation --real   # real embedding and reranking models
```

### Docker

```bash
docker compose up --build   # API on port 8000 plus a Qdrant server on port 6333
```

The image runs as a non-root user, contains no `.env` file, and has a built-in health check.
Vectors persist in the `qdrant_data` volume and the downloaded models in `model_cache`.
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

`POST /search` returns the chunks of that repository that best answer a question:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"repo_url": "https://github.com/yashG72/repomind", "query": "parse_github_repo_url", "top_k": 3}'
```

```json
{
  "repository": "yashG72/repomind",
  "query": "parse_github_repo_url",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "results": [
    {
      "score": 7.359944,
      "path": "src/repomind/ingestion/github_url.py",
      "start_line": 31,
      "end_line": 51,
      "symbol": "def parse_github_repo_url",
      "language": "python",
      "commit_sha": "5711c5613e83c3a565a0c549e7a40ea5fd01720a",
      "url": "https://github.com/yashG72/repomind/blob/5711c5613e83c3a565a0c549e7a40ea5fd01720a/src/repomind/ingestion/github_url.py#L31-L51",
      "text": "def parse_github_repo_url(url: str) -> RepoRef: ...",
      "scores": {
        "semantic": 0.61904025,
        "semantic_rank": 5,
        "keyword": 16.341568,
        "keyword_rank": 1,
        "rrf": 0.031778,
        "rerank": 7.3599443435668945
      }
    }
  ],
  "duration_ms": 805.81,
  "mode": "hybrid",
  "reranker_model": "cross-encoder/ms-marco-MiniLM-L6-v2",
  "warnings": []
}
```

(Abridged from a real run against this repository with the default models and a Qdrant
server. Semantic search alone ranked this function 5th; keyword search ranked it 1st, and
fusion plus reranking put it first. Once the models were loaded, a hybrid search with
reranking took about 0.5 s on a 4-core CPU, against about 20 ms in semantic mode.)

| Request field | Meaning |
|---|---|
| `top_k` | Number of results, 1–50 (default `REPOMIND_SEARCH_DEFAULT_TOP_K`, 5) |
| `commit_sha` | Optional: search one specific indexed commit |
| `mode` | Optional: `"hybrid"` or `"semantic"` (default `REPOMIND_SEARCH_MODE`, hybrid) |
| `rerank` | Optional, hybrid only: `true`/`false` to turn reranking on or off for this request (default `REPOMIND_RERANK_ENABLED`) |

| Response field | Meaning |
|---|---|
| `mode` | The mode that ran |
| `reranker_model` | The reranking model, or `null` if results were not reranked |
| `warnings` | Anything that limited the search, e.g. keyword search unavailable on an old collection |
| `results[].score` | The score that produced the order: cosine similarity in semantic mode; in hybrid mode the reranker score (or the RRF score for results that were not reranked). Compare within one query only |
| `results[].scores` | What each stage said about the result (`semantic`, `keyword`, `rrf`, `rerank` scores and the 1-based `semantic_rank`/`keyword_rank`); `null` where a stage did not run or did not find it. For debugging |

All Phase 3 fields (`score`, `path`, `start_line`, `end_line`, `symbol`, `language`,
`commit_sha`, `url`, `text`) and the citation URL format are unchanged; the new fields are
additions. One behaviour change: in hybrid mode (the new default) `score` is no longer a
cosine similarity. Clients that relied on that should send `"mode": "semantic"` or read
`scores.semantic`.

Search returns 404 if the repository has not been indexed, 503 if a model or Qdrant is
unavailable, and 422 for an empty or too-long query, an unknown mode, or `rerank: true` when
reranking is not possible (semantic mode, or disabled by `REPOMIND_RERANK_ENABLED=false`).

### How hybrid search works

Two kinds of search are good at different questions:

- **Semantic search** compares meanings. "How are passwords stored?" finds `hash_password`
  even though the question never says "hash". It can miss exact names, especially in large
  repositories where many chunks look alike.
- **Keyword search** matches words. `parse_github_repo_url`, `github_fetcher.py`,
  `MAX_UPLOAD_BYTES` or an error message like "Repository archive is above the download
  limit" are found precisely, but a question phrased in different words finds little.

Hybrid mode runs both and merges the results:

1. **Validate** the query and repository, and check the repository (and commit) is indexed.
2. **Semantic search**: embed the question once and take the closest
   `REPOMIND_SEARCH_SEMANTIC_CANDIDATES` chunks (50).
3. **Keyword search**: take the best `REPOMIND_SEARCH_KEYWORD_CANDIDATES` chunks (50) by
   BM25, the standard search-engine ranking formula. Both searches use the same repository and
   commit filter.
4. **Reciprocal Rank Fusion (RRF)** merges the two lists. The two scores are on different
   scales (cosine similarity vs BM25), so RRF uses only positions: each chunk gets
   `1 / (k + rank)` from each list it appears in, with `k = REPOMIND_SEARCH_RRF_K` (60). A
   chunk ranked well by both searches rises to the top. Chunks are matched by their id, so the
   same lines in another repository or commit are never merged.
5. **Limit**: only the top `REPOMIND_RERANK_MAX_CANDIDATES` (20) fused chunks go to the
   reranker, which bounds the CPU time per request.
6. **Rerank**: a cross-encoder model reads the question and each candidate together and
   scores how well the chunk answers it. This is more accurate than comparing vectors but too
   slow for the whole repository, which is why it only sees the shortlist. Candidates beyond
   the shortlist keep their fused order after the reranked ones.
7. **Return** the top `top_k` with the usual path, line numbers and GitHub link.

**How keyword search is stored.** No separate search service is needed. Each chunk's Qdrant
point carries a second, "sparse" vector next to its embedding: a list of the chunk's terms
(from its path, symbol and code) with BM25 weights. Qdrant keeps these in an inverted index
and applies the "rare words count more" (IDF) part itself, so a keyword search only touches
chunks that share a term with the question and nothing is loaded into memory per request.
Tokenization is code-aware: `parseGitHubURL` and `hash_password` are kept whole (for exact
lookups) and also split into `parse`/`git`/`hub`/`url` and `hash`/`password`; file names like
`github_fetcher.py` are kept whole too. Common English question words ("how", "where",
"the") are ignored, and a query uses at most 32 distinct terms. See
`src/repomind/retrieval/keywords.py`.

**Semantic mode** is exactly Phase 3's search (embed, nearest `top_k` chunks, cosine score)
and is kept for backward compatibility and for comparing modes.

### Upgrading from Phase 3

Qdrant cannot add the keyword vector to a collection that already exists, so a collection
created by Phase 3 (for example the `qdrant_data` Docker volume) has no keyword index. With
such a collection everything keeps working, but hybrid search uses semantic results only and
says so in `warnings`. To enable keyword search, either:

- set `REPOMIND_QDRANT_COLLECTION` to a new name (e.g. `repomind_chunks_v2`) and call
  `/index` again for each repository, or
- delete the old collection (or the `data/qdrant` folder in local mode) and re-index.

New collections are created with keyword support automatically.

### Choosing the reranking model

`REPOMIND_RERANK_MODEL` accepts a Hugging Face cross-encoder id or a local model folder. It
uses the same device, cache folder and offline setting as the embedding model
(`REPOMIND_EMBEDDING_DEVICE`, `REPOMIND_EMBEDDING_CACHE_DIR`,
`REPOMIND_EMBEDDING_LOCAL_FILES_ONLY`). Models that need `trust_remote_code` are refused,
because that would run Python code downloaded with the model. The model loads on the first
hybrid search and is then reused; if it cannot be downloaded or loaded, hybrid searches that
rerank return 503 with a clear message, and `"rerank": false`, `"mode": "semantic"` or
`REPOMIND_RERANK_ENABLED=false` keep search working without it.

Measured on a 4-core CPU, reranking 20 candidates of up to 512 tokens, with the retrieval
evaluation's realistic dataset (RepoMind's own code, 19 judged questions):

| Model | Size | Rerank 20 candidates | MRR (hybrid + rerank) | Notes |
|---|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L6-v2` (default) | 23M params | ~0.5 s | 0.675 | Fast and small. Trained on web search, not code |
| `mixedbread-ai/mxbai-rerank-xsmall-v1` | 71M params | ~3.2 s | 0.725 | Best quality here, about 6x slower |
| `BAAI/bge-reranker-base` | 278M params | ~2.8 s | 0.714 | Larger download and memory |
| `cross-encoder/ms-marco-MiniLM-L12-v2` | 33M params | not timed | 0.668 | No gain over L6 here |

For comparison, hybrid without reranking scored 0.680 and semantic-only 0.574 on the same
questions. The default favours speed on a CPU: with the default model, reranking did not
improve MRR on the realistic dataset (it helped on the small synthetic one). These numbers
come from a small dataset and the larger models were picked using it, so treat them as a
guide, not a benchmark.

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

- Keyword IDF (how rare a term is) is computed by Qdrant across the whole collection, i.e.
  over every indexed repository, not per repository. Results are still filtered to one
  repository; only the weighting of terms is shared.
- Keyword term ids are 32-bit hashes, so two different terms can rarely share an id and add
  a little noise to each other's scores.
- The default reranker was trained on web search, not code. It is a modest help at best on
  code questions (see the evaluation); a larger reranker is more accurate but slower on a CPU.
- Reranker scores (logits) are not calibrated: a low score does not reliably mean "no
  answer". Every search returns its top results even when nothing in the repository is
  relevant.
- `all-MiniLM-L6-v2` reads about the first 256 tokens of each chunk and ignores the rest,
  so the end of a long chunk does not influence its vector (keyword search does see the
  whole chunk; the reranker reads about the first 512 tokens).
- MiniLM is a general English text model, not a code-specific one; it works best for
  questions phrased like the code's names and docstrings.
- Indexing runs inside the request and on the CPU by default; large repositories take a while.
- If indexing is interrupted after the new commit is stored but before old chunks are
  deleted, both versions can briefly appear in results; re-running `/index` fixes it.

## Evaluation

`tests/evaluation/` holds a small retrieval evaluation that compares semantic-only, hybrid
(RRF) and hybrid + rerank search. Questions and their relevant code were written by reading
the code before running any search. Two datasets:

- **Synthetic shop + blog** (24 questions): a small invented `octo/shop` backend, plus
  `octo/blog` indexed into the same collection with deliberately overlapping names, to check
  repository isolation.
- **RepoMind @ 5711c56** (21 questions): RepoMind's own code at the Phase 3 merge (about 320
  chunks), read from this git repository, so it is realistic and repeatable.

Questions cover exact function names, exact class/identifier names, conceptual questions,
questions where keyword and semantic rankings disagree (error messages, file names, a name
inside a sentence), repository isolation, and questions with no answer in the repository.
Reported metrics: Recall@1/3/5, MRR (mean of 1 / rank of the first relevant result) and the
rank of the first relevant result per question; isolation is checked on every question in
every mode (results from another repository count as "leaks").

`uv run pytest` runs the harness with deterministic fakes (it checks metrics and isolation,
not which mode wins). The real-model numbers need the models downloaded:
`uv run python -m tests.evaluation --real`. Latest results:
[`docs/retrieval-evaluation.md`](docs/retrieval-evaluation.md).

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
| Hybrid search | Keyword (BM25) search, fused with semantic search by RRF | 4 ✅ |
| Reranker | Local cross-encoder reorders the best candidates | 4 ✅ |
| Context builder | Pick top chunks within a token budget, label them with citations | later |
| LLM client | Provider-agnostic text generation | later |
| API | FastAPI endpoints (`/health`, `/ingest`, `/index`, `/search`) | 1–4 ✅ |

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
  vectorstore/qdrant.py      VectorStore: collection setup, upsert, scoped semantic + keyword search
  retrieval/indexing.py      Ingest -> embed (only new chunks) -> store -> drop old commits
  retrieval/keywords.py      Code-aware tokenizer and BM25 term weights (pure functions)
  retrieval/fusion.py        Reciprocal Rank Fusion (pure function)
  retrieval/search.py        The retrieval pipeline: semantic or hybrid (+ rerank)
  reranking/cross_encoder.py Reranker: loads the cross-encoder once, validates its scores
tests/                       Unit tests; GitHub, the models and Qdrant are faked or embedded
tests/integration/           Real models + real Qdrant (opt-in, see above)
tests/evaluation/            Retrieval evaluation dataset, metrics and report
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
