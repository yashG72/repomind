"""The code-aware keyword tokenizer and BM25 encoding (pure functions, no Qdrant)."""

from repomind.retrieval.keywords import (
    MAX_QUERY_TERMS,
    encode_document,
    encode_query,
    term_id,
    tokenize,
)


def test_snake_case_is_kept_whole_and_split() -> None:
    assert tokenize("hash_password") == ["hash_password", "hash", "password"]


def test_camel_case_and_acronyms_are_split() -> None:
    assert tokenize("parseGitHubURL") == ["parsegithuburl", "parse", "git", "hub", "url"]
    assert tokenize("HTTPServer") == ["httpserver", "http", "server"]


def test_file_names_and_dotted_names_are_kept_whole() -> None:
    terms = tokenize("src/repomind/ingestion/github_fetcher.py")

    assert "github_fetcher.py" in terms
    assert {"github_fetcher", "github", "fetcher", "py", "ingestion", "repomind"} <= set(terms)
    assert "os.path.join" in tokenize("os.path.join(a, b)")


def test_error_messages_become_words() -> None:
    assert tokenize('raise ConnectionError("gave up")') == [
        "raise",
        "connectionerror",
        "connection",
        "error",
        "gave",
        "up",
    ]


def test_stopwords_single_characters_and_huge_tokens_are_dropped() -> None:
    assert tokenize("Where is the x?") == []
    assert tokenize("a" * 65) == []
    assert tokenize("a" * 64) == ["a" * 64]


def test_term_ids_are_stable_unsigned_32_bit() -> None:
    assert term_id("hash_password") == term_id("hash_password")
    assert term_id("hash") != term_id("password")
    assert 0 <= term_id("anything") < 2**32


def test_document_weights_saturate_with_repetition() -> None:
    once = encode_document("retry")
    many = encode_document("retry " * 50)

    assert once.indices == many.indices == [term_id("retry")]
    assert many.values[0] > 0
    # BM25 saturation: 50 repeats add far less than 50x the weight (k1 + 1 = 2.2 at most).
    assert many.values[0] < 2.2


def test_longer_documents_get_lower_weight_per_term() -> None:
    short = encode_document("retry backoff")
    long = encode_document("retry backoff " + " ".join(f"word{i}" for i in range(300)))

    weight = dict(zip(short.indices, short.values, strict=True))[term_id("retry")]
    long_weight = dict(zip(long.indices, long.values, strict=True))[term_id("retry")]
    assert long_weight < weight


def test_query_terms_are_unique_weight_one_and_bounded() -> None:
    query = encode_query("hash hash password")
    assert sorted(query.indices) == query.indices
    assert query.values == [1.0, 1.0]

    long_query = encode_query(" ".join(f"term{i}" for i in range(100)))
    assert len(long_query.indices) == MAX_QUERY_TERMS


def test_query_with_nothing_searchable_is_empty() -> None:
    assert not encode_query("?? !! the is a")
