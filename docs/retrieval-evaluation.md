# Retrieval evaluation results

Generated with `uv run python -m tests.evaluation` (fakes) and
`uv run python -m tests.evaluation --real --qdrant-url http://localhost:6333` (real models), on
2026-10-25, branch `phase-4-hybrid-search`, Qdrant server 1.20.0, CPU only. The real-model run
gave identical numbers with the embedded Qdrant engine. Datasets and metrics are described in
the README's Evaluation section and in `tests/evaluation/`.

## Summary

- **Hybrid (RRF) vs semantic.** On the realistic RepoMind dataset hybrid search raised MRR
  from 0.574 to 0.680 and Recall@1 from 0.421 to 0.579. The gains are on exact names and
  keyword-heavy questions (semantic -> hybrid -> hybrid + rerank: `SkipReason` not found ->
  8 -> 1; `parse_github_repo_url` 5 -> 2 -> 1; `minified file detection threshold`
  5 -> 1 -> 2). On the small synthetic dataset, where semantic
  search was already near perfect, hybrid was slightly *worse* (MRR 0.952 -> 0.914): keyword
  matches on common words pushed two conceptual answers down (`How are old logins cleaned
  up?`: rank 1 -> 5; `How are user credentials stored securely?`: 1 -> 2).
- **Reranking (default model `cross-encoder/ms-marco-MiniLM-L6-v2`).** Mixed. On the
  synthetic set it fixed the hybrid regressions (MRR 0.976, best of the three). On the
  realistic set MRR was flat (0.680 -> 0.675): Recall@5 rose (0.763 -> 0.816) but Recall@1
  fell (0.579 -> 0.526), with some exact lookups dropping from rank 1 to 2 (`redact_secrets`,
  `LOCK_FILES`) and one file-name question from 1 to 3. The model was trained on web search,
  not code. Larger rerankers did better on the realistic set
  (`mixedbread-ai/mxbai-rerank-xsmall-v1` 0.725, `BAAI/bge-reranker-base` 0.714) but took
  about 3 s instead of 0.5 s to rerank 20 candidates on a 4-core CPU.
- **Isolation.** No results from another repository in any mode, on any question (0 leaks).
- **No-match questions** still return top results (search has no relevance threshold), and
  reranker scores are not calibrated enough to detect "no answer".
- Both datasets are small (21 and 19 judged questions); one question moving changes MRR by
  about 0.03-0.05. The larger reranker models were compared on this same dataset.

## Real models: all-MiniLM-L6-v2 + ms-marco-MiniLM-L6-v2

### Synthetic shop + blog

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Leaks |
|---|---|---|---|---|---|
| semantic | 0.905 | 1.000 | 1.000 | 0.952 | 0 |
| hybrid (RRF) | 0.857 | 0.952 | 1.000 | 0.914 | 0 |
| hybrid + rerank | 0.929 | 1.000 | 1.000 | 0.976 | 0 |

21 questions have relevant code; 3 have none (no-match and isolation questions are checked for leaks only). Ranks below are of the first relevant result in the top 10; '-' means not found.

| Question | Category | semantic | hybrid (RRF) | hybrid + rerank |
|---|---|---|---|---|
| `calculate_invoice_total` | exact function | 1 | 1 | 1 |
| `where is retry_with_backoff defined?` | exact function | 1 | 1 | 1 |
| `purge_expired` | exact function | 1 | 1 | 1 |
| `addToCart` | exact function | 1 | 1 | 1 |
| `InvoiceNotFoundError` | exact identifier | 1 | 1 | 1 |
| `What is MAX_UPLOAD_BYTES set to?` | exact identifier | 1 | 1 | 1 |
| `PaymentGateway` | exact identifier | 1 | 1 | 1 |
| `SHOP_TAX_RATE` | exact identifier | 1 | 1 | 1 |
| `How are user credentials stored securely?` | conceptual | 1 | 2 | 1 |
| `How is a login attempt checked against what was saved?` | conceptual | 1 | 1 | 1 |
| `How does the app avoid doing too many operations per second?` | conceptual | 1 | 1 | 1 |
| `What happens when the connection to a remote service keeps failing?` | conceptual | 1 | 1 | 1 |
| `How are old logins cleaned up?` | conceptual | 1 | 5 | 2 |
| `How is tax added to the bill?` | conceptual | 1 | 1 | 1 |
| `payment provider declined the card` | keyword vs semantic | 2 | 1 | 1 |
| `what is in webhooks.py` | keyword vs semantic | 2 | 2 | 1 |
| `gave up after repeated connection failures` | keyword vs semantic | 1 | 1 | 1 |
| `how does verify_signature protect the webhook endpoint from forged requests` | keyword vs semantic | 1 | 1 | 1 |
| `promo code percentage off` | keyword vs semantic | 1 | 1 | 1 |
| `hash_password bcrypt` | repository isolation | 1 | 1 | 1 |
| `RateLimiter fixed window per minute` | repository isolation | 1 | 1 | 1 |
| `render_rss_feed` | repository isolation | n/a (top: README.md) | n/a (top: README.md) | n/a (top: shop/storage/uploads.py) |
| `How is the GraphQL schema stitched together?` | no match | n/a (top: README.md) | n/a (top: README.md) | n/a (top: shop/net/rate_limit.py class RateLimiter) |
| `kubernetes helm chart values` | no match | n/a (top: web/cart.js) | n/a (top: web/cart.js) | n/a (top: shop/net/retry.py) |

### RepoMind @ 5711c56

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Leaks |
|---|---|---|---|---|---|
| semantic | 0.421 | 0.658 | 0.763 | 0.574 | 0 |
| hybrid (RRF) | 0.579 | 0.763 | 0.763 | 0.680 | 0 |
| hybrid + rerank | 0.526 | 0.763 | 0.816 | 0.675 | 0 |

19 questions have relevant code; 2 have none (no-match and isolation questions are checked for leaks only). Ranks below are of the first relevant result in the top 10; '-' means not found.

| Question | Category | semantic | hybrid (RRF) | hybrid + rerank |
|---|---|---|---|---|
| `parse_github_repo_url` | exact function | 5 | 2 | 1 |
| `redact_secrets` | exact function | 2 | 1 | 2 |
| `where is validate_vectors defined?` | exact function | 1 | 1 | 1 |
| `delete_other_commits` | exact function | 1 | 1 | 1 |
| `RepositoryTooLarge` | exact identifier | 1 | 1 | 1 |
| `_STATUS_BY_ERROR` | exact identifier | 1 | 1 | 1 |
| `LOCK_FILES` | exact identifier | 1 | 1 | 2 |
| `SkipReason` | exact identifier | - | 8 | 1 |
| `How does the app stop a malicious archive from writing files outside the repo?` | conceptual | 2 | 3 | 2 |
| `Which HTTP status code does the API return when GitHub rate limits us?` | conceptual | - | 8 | 9 |
| `Why are embeddings not recomputed when the same repository is indexed again?` | conceptual | - | - | 8 |
| `How are Python files split into pieces?` | conceptual | 1 | 1 | 1 |
| `How is the model loaded only once when many requests arrive at the same time?` | conceptual | - | - | - |
| `How are log lines written as JSON?` | conceptual | 2 | 3 | 4 |
| `Repository archive is above the download limit` | keyword vs semantic | 1 | 1 | 1 |
| `GitHub redirected to a destination that is not allowed` | keyword vs semantic | 2 | 2 | 1 |
| `what does github_url.py accept` | keyword vs semantic | 2 | 1 | 3 |
| `how does _looks_rate_limited decide` | keyword vs semantic | 1 | 1 | 1 |
| `minified file detection threshold` | keyword vs semantic | 5 | 1 | 2 |
| `How are password reset emails sent?` | no match | n/a (top: src/repomind/ingestion/secrets.py def redact_secrets) | n/a (top: src/repomind/ingestion/github_fetcher.py def _retry_after_seconds) | n/a (top: src/repomind/ingestion/github_fetcher.py def GitHubFetcher._send) |
| `Where is the GraphQL resolver for users?` | no match | n/a (top: src/repomind/api/dependencies.py def get_indexing_service) | n/a (top: src/repomind/api/errors.py) | n/a (top: src/repomind/api/errors.py) |


## Deterministic fakes (unit-test configuration)

These numbers use `FakeEmbedder` (a hashed bag of words, not a language model) and
`FakeReranker` (counts shared words). They show the harness works and is repeatable; they say
nothing about real semantic search quality.

### Synthetic shop + blog

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Leaks |
|---|---|---|---|---|---|
| semantic | 0.333 | 0.524 | 0.619 | 0.461 | 0 |
| hybrid (RRF) | 0.667 | 0.929 | 0.929 | 0.802 | 0 |
| hybrid + rerank | 0.786 | 0.952 | 0.952 | 0.873 | 0 |

21 questions have relevant code; 3 have none (no-match and isolation questions are checked for leaks only). Ranks below are of the first relevant result in the top 10; '-' means not found.

| Question | Category | semantic | hybrid (RRF) | hybrid + rerank |
|---|---|---|---|---|
| `calculate_invoice_total` | exact function | 1 | 1 | 1 |
| `where is retry_with_backoff defined?` | exact function | 3 | 2 | 2 |
| `purge_expired` | exact function | 6 | 1 | 1 |
| `addToCart` | exact function | 5 | 1 | 1 |
| `InvoiceNotFoundError` | exact identifier | 1 | 1 | 1 |
| `What is MAX_UPLOAD_BYTES set to?` | exact identifier | 1 | 1 | 1 |
| `PaymentGateway` | exact identifier | 3 | 1 | 1 |
| `SHOP_TAX_RATE` | exact identifier | 2 | 1 | 1 |
| `How are user credentials stored securely?` | conceptual | 9 | 3 | 3 |
| `How is a login attempt checked against what was saved?` | conceptual | 10 | 2 | 1 |
| `How does the app avoid doing too many operations per second?` | conceptual | - | 1 | 1 |
| `What happens when the connection to a remote service keeps failing?` | conceptual | 5 | 2 | 1 |
| `How are old logins cleaned up?` | conceptual | - | - | - |
| `How is tax added to the bill?` | conceptual | 9 | 2 | 2 |
| `payment provider declined the card` | keyword vs semantic | 2 | 1 | 1 |
| `what is in webhooks.py` | keyword vs semantic | 8 | 2 | 1 |
| `gave up after repeated connection failures` | keyword vs semantic | 1 | 1 | 1 |
| `how does verify_signature protect the webhook endpoint from forged requests` | keyword vs semantic | 1 | 1 | 1 |
| `promo code percentage off` | keyword vs semantic | 1 | 1 | 1 |
| `hash_password bcrypt` | repository isolation | 1 | 1 | 1 |
| `RateLimiter fixed window per minute` | repository isolation | - | 1 | 1 |
| `render_rss_feed` | repository isolation | n/a (top: shop/net/rate_limit.py class RateLimiter) | n/a (top: shop/net/rate_limit.py class RateLimiter) | n/a (top: shop/net/rate_limit.py class RateLimiter) |
| `How is the GraphQL schema stitched together?` | no match | n/a (top: shop/billing/payments.py) | n/a (top: shop/billing/payments.py) | n/a (top: shop/billing/payments.py) |
| `kubernetes helm chart values` | no match | n/a (top: shop/billing/invoice.py def load_invoice) | n/a (top: shop/billing/invoice.py def load_invoice) | n/a (top: shop/billing/invoice.py def load_invoice) |

### RepoMind @ 5711c56

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Leaks |
|---|---|---|---|---|---|
| semantic | 0.105 | 0.368 | 0.421 | 0.255 | 0 |
| hybrid (RRF) | 0.421 | 0.684 | 0.711 | 0.547 | 0 |
| hybrid + rerank | 0.526 | 0.711 | 0.711 | 0.622 | 0 |

19 questions have relevant code; 2 have none (no-match and isolation questions are checked for leaks only). Ranks below are of the first relevant result in the top 10; '-' means not found.

| Question | Category | semantic | hybrid (RRF) | hybrid + rerank |
|---|---|---|---|---|
| `parse_github_repo_url` | exact function | 7 | 3 | 3 |
| `redact_secrets` | exact function | 8 | 2 | 2 |
| `where is validate_vectors defined?` | exact function | - | 3 | 1 |
| `delete_other_commits` | exact function | 2 | 1 | 1 |
| `RepositoryTooLarge` | exact identifier | 1 | 1 | 1 |
| `_STATUS_BY_ERROR` | exact identifier | 2 | 1 | 1 |
| `LOCK_FILES` | exact identifier | 2 | 1 | 1 |
| `SkipReason` | exact identifier | - | 7 | 7 |
| `How does the app stop a malicious archive from writing files outside the repo?` | conceptual | - | - | - |
| `Which HTTP status code does the API return when GitHub rate limits us?` | conceptual | 3 | 1 | 1 |
| `Why are embeddings not recomputed when the same repository is indexed again?` | conceptual | - | 4 | 3 |
| `How are Python files split into pieces?` | conceptual | 6 | 1 | 1 |
| `How is the model loaded only once when many requests arrive at the same time?` | conceptual | - | - | - |
| `How are log lines written as JSON?` | conceptual | - | - | - |
| `Repository archive is above the download limit` | keyword vs semantic | 3 | 2 | 1 |
| `GitHub redirected to a destination that is not allowed` | keyword vs semantic | 1 | 1 | 1 |
| `what does github_url.py accept` | keyword vs semantic | - | 3 | 2 |
| `how does _looks_rate_limited decide` | keyword vs semantic | 4 | 1 | 1 |
| `minified file detection threshold` | keyword vs semantic | - | - | - |
| `How are password reset emails sent?` | no match | n/a (top: tests/test_config.py def test_secrets_are_masked_in_repr) | n/a (top: tests/test_indexing.py) | n/a (top: tests/test_indexing.py) |
| `Where is the GraphQL resolver for users?` | no match | n/a (top: src/repomind/ingestion/chunking.py) | n/a (top: src/repomind/ingestion/file_filter.py) | n/a (top: src/repomind/ingestion/file_filter.py) |

