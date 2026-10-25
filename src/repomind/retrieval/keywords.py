"""Keyword (lexical) encoding for code: turns text into weighted terms for BM25-style search.

Semantic search finds code that *means* something similar to the question, but it can miss
exact names: a question about `hash_password` or `github_fetcher.py` should find the code
that contains exactly those words. Keyword search covers that case.

Each chunk becomes a "sparse vector": a short list of (term id, weight) pairs, one for every
distinct term in the chunk's path, symbol and code. Qdrant stores these in an inverted index
(like a book's index: term -> chunks containing it), so a search only touches chunks that
share a term with the question. The weights follow BM25, the classic search-engine formula:
- term frequency, with saturation: a word appearing 10 times is not 10x as relevant as once;
- length normalization: long chunks do not win just by containing more words;
- inverse document frequency (rare terms count more): computed by Qdrant itself across the
  stored chunks (the collection's sparse vector uses Qdrant's IDF modifier).

Code-aware tokenization: `parseGitHubURL`, `hash_password` and `github_fetcher.py` are kept
whole (for exact lookups) and also split into their parts (`parse`, `git`, `hub`, `url`,
`hash`, `password`, `github`, `fetcher`, `py`), so both exact names and partial words match.

Everything here is pure Python: no model, no network, deterministic.
"""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

# Bump when tokenization or weighting changes: chunks stored with an older version are
# re-encoded on the next /index, and searches only use chunks with the current version.
KEYWORD_ENCODING_VERSION = 1

# BM25 parameters. k1 controls how quickly repeated terms stop adding weight; b controls
# length normalization. AVERAGE_TERMS is a fixed estimate of a chunk's length in terms
# (real BM25 uses the corpus average; a constant keeps documents independent of each other).
BM25_K1 = 1.2
BM25_B = 0.75
AVERAGE_TERMS = 150.0

MAX_TERM_CHARS = 64  # longer "words" (hashes, base64 blobs) are not useful search terms
MAX_QUERY_TERMS = 32  # bounds the work a single query can ask Qdrant to do

# A run of identifier characters, optionally joined by "." or "-" (file names, dotted paths,
# kebab-case): `github_fetcher.py`, `os.path.join`, `max-age`.
_COMPOUND = re.compile(r"[A-Za-z0-9_]+(?:[.\-][A-Za-z0-9_]+)*")
_IDENTIFIER = re.compile(r"[A-Za-z0-9_]+")
# camelCase / PascalCase boundaries, keeping acronyms together: "parseGitHubURL" ->
# "parse", "Git", "Hub", "URL"; "HTTPServer" -> "HTTP", "Server".
_CAMEL_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

# Words that carry no meaning in a question about code. IDF already lowers the weight of
# common words; dropping these also keeps them from using the query-term budget.
_STOPWORDS = frozenset(
    """
    a an and any are as at be by can do does for from has have how i if in into is it its
    me my of on or our should so that the their them then there these this those to us was
    we what when where which who why will with you your
    """.split()
)


@dataclass(frozen=True)
class SparseTerms:
    """A sparse vector: parallel lists of term ids and weights, ids sorted ascending."""

    indices: list[int]
    values: list[float]

    def __bool__(self) -> bool:
        return bool(self.indices)


def tokenize(text: str) -> list[str]:
    """Lower-cased search terms in order of appearance (with repeats), code-aware."""
    terms: list[str] = []
    for compound in _COMPOUND.findall(text):
        if "." in compound or "-" in compound:
            _add(terms, compound.lower())
        for identifier in _IDENTIFIER.findall(compound):
            _add(terms, identifier.lower())
            parts = [p for piece in identifier.split("_") for p in _CAMEL_PART.findall(piece)]
            if len(parts) > 1:
                for part in parts:
                    _add(terms, part.lower())
    return terms


def term_id(term: str) -> int:
    """A stable 32-bit id for a term (Qdrant sparse indices are unsigned 32-bit integers).

    Hashing avoids keeping a vocabulary table. Two different terms can share an id, but
    with 4 billion ids that is rare and only adds a little noise to one term's score.
    """
    return int.from_bytes(hashlib.blake2b(term.encode(), digest_size=4).digest(), "big")


def encode_document(text: str) -> SparseTerms:
    """BM25 term weights for one chunk (term frequency with saturation and length norm)."""
    counts = Counter(tokenize(text))
    length = sum(counts.values())
    norm = BM25_K1 * (1 - BM25_B + BM25_B * length / AVERAGE_TERMS)
    weights: dict[int, float] = {}
    for term, tf in counts.items():
        key = term_id(term)
        weights[key] = weights.get(key, 0.0) + tf * (BM25_K1 + 1) / (tf + norm)
    return _sparse(weights)


def encode_query(text: str) -> SparseTerms:
    """Each distinct query term with weight 1; Qdrant multiplies in the term's IDF."""
    unique = list(dict.fromkeys(tokenize(text)))[:MAX_QUERY_TERMS]
    return _sparse(dict.fromkeys((term_id(t) for t in unique), 1.0))


def _add(terms: list[str], term: str) -> None:
    if 1 < len(term) <= MAX_TERM_CHARS and term not in _STOPWORDS:
        terms.append(term)


def _sparse(weights: dict[int, float]) -> SparseTerms:
    indices = sorted(weights)
    return SparseTerms(indices=indices, values=[weights[i] for i in indices])
