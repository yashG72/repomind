"""Evaluation questions with known relevant code, for `octo/shop` (see corpus.py).

A result counts as relevant when its path matches and, if a symbol is given, its symbol
matches too. Relevance was decided by reading the corpus, before any retrieval was run.
"""

from dataclasses import dataclass, field

SHOP_URL = "https://github.com/octo/shop"
BLOG_URL = "https://github.com/octo/blog"


@dataclass(frozen=True)
class EvalQuery:
    id: str
    category: str
    question: str
    # (path, symbol or None). Empty for questions with no useful match in the repository.
    relevant: tuple[tuple[str, str | None], ...] = field(default_factory=tuple)
    repo_url: str = SHOP_URL


QUERIES: tuple[EvalQuery, ...] = (
    # Exact function-name lookups.
    EvalQuery(
        "fn-invoice-total",
        "exact function",
        "calculate_invoice_total",
        (("shop/billing/invoice.py", "def calculate_invoice_total"),),
    ),
    EvalQuery(
        "fn-retry",
        "exact function",
        "where is retry_with_backoff defined?",
        (("shop/net/retry.py", "def retry_with_backoff"),),
    ),
    EvalQuery(
        "fn-purge",
        "exact function",
        "purge_expired",
        (("shop/auth/sessions.py", "class SessionStore"),),
    ),
    EvalQuery(
        "fn-js-cart",
        "exact function",
        "addToCart",
        (("web/cart.js", None),),
    ),
    # Exact class names and identifiers.
    EvalQuery(
        "id-invoice-not-found",
        "exact identifier",
        "InvoiceNotFoundError",
        (("shop/billing/errors.py", "class InvoiceNotFoundError"),),
    ),
    EvalQuery(
        "id-max-upload",
        "exact identifier",
        "What is MAX_UPLOAD_BYTES set to?",
        (("shop/storage/uploads.py", None),),
    ),
    EvalQuery(
        "id-gateway",
        "exact identifier",
        "PaymentGateway",
        (("shop/billing/payments.py", "class PaymentGateway"),),
    ),
    EvalQuery(
        "id-env-var",
        "exact identifier",
        "SHOP_TAX_RATE",
        (("shop/config.py", None),),
    ),
    # Conceptual questions that do not use the code's names.
    EvalQuery(
        "concept-passwords",
        "conceptual",
        "How are user credentials stored securely?",
        (("shop/auth/passwords.py", "def hash_password"),),
    ),
    EvalQuery(
        "concept-login-check",
        "conceptual",
        "How is a login attempt checked against what was saved?",
        (("shop/auth/passwords.py", "def verify_password"),),
    ),
    EvalQuery(
        "concept-throttle",
        "conceptual",
        "How does the app avoid doing too many operations per second?",
        (("shop/net/rate_limit.py", "class RateLimiter"),),
    ),
    EvalQuery(
        "concept-flaky-network",
        "conceptual",
        "What happens when the connection to a remote service keeps failing?",
        (("shop/net/retry.py", "def retry_with_backoff"),),
    ),
    EvalQuery(
        "concept-logout",
        "conceptual",
        "How are old logins cleaned up?",
        (("shop/auth/sessions.py", "class SessionStore"),),
    ),
    EvalQuery(
        "concept-tax",
        "conceptual",
        "How is tax added to the bill?",
        (("shop/billing/invoice.py", "def calculate_invoice_total"),),
    ),
    # Keyword and semantic rankings are expected to disagree: an error message, a file name,
    # a function named in a sentence, and a question mixing a name with a concept.
    EvalQuery(
        "mixed-error-message",
        "keyword vs semantic",
        "payment provider declined the card",
        (("shop/billing/payments.py", "class PaymentGateway"),),
    ),
    EvalQuery(
        "mixed-filename",
        "keyword vs semantic",
        "what is in webhooks.py",
        (
            ("shop/api/webhooks.py", "def verify_signature"),
            ("shop/api/webhooks.py", "def handle_payment_webhook"),
        ),
    ),
    EvalQuery(
        "mixed-gave-up",
        "keyword vs semantic",
        "gave up after repeated connection failures",
        (("shop/net/retry.py", "def retry_with_backoff"),),
    ),
    EvalQuery(
        "mixed-signature",
        "keyword vs semantic",
        "how does verify_signature protect the webhook endpoint from forged requests",
        (("shop/api/webhooks.py", "def verify_signature"),),
    ),
    EvalQuery(
        "mixed-discount",
        "keyword vs semantic",
        "promo code percentage off",
        (("shop/billing/invoice.py", "def apply_discount_code"),),
    ),
    # Repository isolation: these names also exist in octo/blog (indexed in the same
    # collection). Only octo/shop results may come back.
    EvalQuery(
        "iso-hash-password",
        "repository isolation",
        "hash_password bcrypt",
        (("shop/auth/passwords.py", "def hash_password"),),
    ),
    EvalQuery(
        "iso-rate-limiter",
        "repository isolation",
        "RateLimiter fixed window per minute",
        (("shop/net/rate_limit.py", "class RateLimiter"),),
    ),
    EvalQuery(
        "iso-blog-only",
        "repository isolation",
        "render_rss_feed",  # exists only in octo/blog
        (),
    ),
    # No useful match exists in the repository.
    EvalQuery("none-graphql", "no match", "How is the GraphQL schema stitched together?", ()),
    EvalQuery("none-kubernetes", "no match", "kubernetes helm chart values", ()),
)


REPOMIND_URL = "https://github.com/yashG72/repomind"
_F = "src/repomind/"

# Questions about RepoMind's own code at REPOMIND_SNAPSHOT_COMMIT (see corpus.py).
REPOMIND_QUERIES: tuple[EvalQuery, ...] = tuple(
    EvalQuery(qid, category, question, relevant, repo_url=REPOMIND_URL)
    for qid, category, question, relevant in (
        (
            "rm-fn-parse-url",
            "exact function",
            "parse_github_repo_url",
            ((_F + "ingestion/github_url.py", "def parse_github_repo_url"),),
        ),
        (
            "rm-fn-redact",
            "exact function",
            "redact_secrets",
            ((_F + "ingestion/secrets.py", "def redact_secrets"),),
        ),
        (
            "rm-fn-validate-vectors",
            "exact function",
            "where is validate_vectors defined?",
            ((_F + "embeddings/sentence_transformer.py", "def validate_vectors"),),
        ),
        (
            "rm-fn-delete-other-commits",
            "exact function",
            "delete_other_commits",
            ((_F + "vectorstore/qdrant.py", "def QdrantVectorStore.delete_other_commits"),),
        ),
        (
            "rm-id-too-large",
            "exact identifier",
            "RepositoryTooLarge",
            ((_F + "ingestion/errors.py", "class RepositoryTooLarge"),),
        ),
        (
            "rm-id-status-map",
            "exact identifier",
            "_STATUS_BY_ERROR",
            ((_F + "api/errors.py", None),),
        ),
        (
            "rm-id-lock-files",
            "exact identifier",
            "LOCK_FILES",
            ((_F + "ingestion/file_filter.py", None),),
        ),
        (
            "rm-id-skip-reason",
            "exact identifier",
            "SkipReason",
            ((_F + "core/models.py", "class SkipReason"),),
        ),
        (
            "rm-concept-archive-escape",
            "conceptual",
            "How does the app stop a malicious archive from writing files outside the repo?",
            (
                (_F + "ingestion/github_fetcher.py", "def GitHubFetcher._read_archive"),
                (_F + "ingestion/file_filter.py", "def normalize_archive_path"),
            ),
        ),
        (
            "rm-concept-rate-limit-status",
            "conceptual",
            "Which HTTP status code does the API return when GitHub rate limits us?",
            ((_F + "api/errors.py", None),),
        ),
        (
            "rm-concept-reuse",
            "conceptual",
            "Why are embeddings not recomputed when the same repository is indexed again?",
            (
                (_F + "retrieval/indexing.py", "class IndexingService"),
                (_F + "vectorstore/qdrant.py", "def QdrantVectorStore.existing_ids"),
            ),
        ),
        (
            "rm-concept-python-chunks",
            "conceptual",
            "How are Python files split into pieces?",
            ((_F + "ingestion/chunking.py", None),),
        ),
        (
            "rm-concept-load-once",
            "conceptual",
            "How is the model loaded only once when many requests arrive at the same time?",
            ((_F + "embeddings/sentence_transformer.py", "def SentenceTransformerEmbedder._load"),),
        ),
        (
            "rm-concept-json-logs",
            "conceptual",
            "How are log lines written as JSON?",
            ((_F + "logging_config.py", "class JsonFormatter"),),
        ),
        (
            "rm-mixed-download-limit",
            "keyword vs semantic",
            "Repository archive is above the download limit",
            ((_F + "ingestion/github_fetcher.py", "def GitHubFetcher._download_archive"),),
        ),
        (
            "rm-mixed-redirect",
            "keyword vs semantic",
            "GitHub redirected to a destination that is not allowed",
            ((_F + "ingestion/github_fetcher.py", "def _validate_destination"),),
        ),
        (
            "rm-mixed-filename",
            "keyword vs semantic",
            "what does github_url.py accept",
            ((_F + "ingestion/github_url.py", None),),
        ),
        (
            "rm-mixed-looks-rate-limited",
            "keyword vs semantic",
            "how does _looks_rate_limited decide",
            ((_F + "ingestion/github_fetcher.py", "def _looks_rate_limited"),),
        ),
        (
            "rm-mixed-minified",
            "keyword vs semantic",
            "minified file detection threshold",
            ((_F + "ingestion/file_filter.py", None),),
        ),
        ("rm-none-email", "no match", "How are password reset emails sent?", ()),
        ("rm-none-graphql", "no match", "Where is the GraphQL resolver for users?", ()),
    )
)
