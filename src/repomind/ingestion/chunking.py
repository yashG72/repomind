"""Splits source files into chunks with exact, 1-based line ranges.

Two strategies:
- Python files are split along their structure using the standard-library `ast` module:
  one chunk per top-level function or class (large classes are split per method), plus
  chunks for the module-level code in between. `ast.parse` only reads the code; it never
  runs it.
- Every other file, and any Python file that cannot be parsed, uses a line-based splitter
  that prefers to break at blank lines.

Guarantees, checked by the tests:
- A chunk's text is exactly lines start_line..end_line of the original file.
- Chunks never overlap, and every non-blank line belongs to exactly one chunk.
- No chunk is longer than chunk_max_lines (or, except for a single very long line,
  chunk_max_chars).

Line numbering matches GitHub's: lines are separated by "\\n", and a trailing "\\r" from
Windows line endings is kept inside the line, so numbering is unaffected.
"""

import ast
import hashlib
import logging
from dataclasses import dataclass

from repomind.core.models import CodeChunk, RepoRef, SourceFile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Span:
    start: int  # 1-based, inclusive
    end: int  # 1-based, inclusive
    symbol: str | None = None


class CodeChunker:
    """Implements the Chunker protocol."""

    def __init__(self, max_lines: int = 80, max_chars: int = 6_000) -> None:
        self.max_lines = max_lines
        self.max_chars = max_chars

    def chunk(self, repo: RepoRef, commit_sha: str, file: SourceFile) -> list[CodeChunk]:
        lines = file.content.split("\n")
        spans: list[_Span] | None = None
        if file.language == "python":
            spans = self._python_spans(file, lines)
        if spans is None:
            spans = [_Span(1, len(lines))]

        chunks: list[CodeChunk] = []
        for span in spans:
            for piece in self._split_to_limits(lines, span):
                trimmed = _trim_blank_edges(lines, piece)
                if trimmed is None:
                    continue
                text = "\n".join(lines[trimmed.start - 1 : trimmed.end])
                chunks.append(
                    CodeChunk(
                        chunk_id=_chunk_id(repo, commit_sha, file.path, trimmed),
                        repo=repo,
                        commit_sha=commit_sha,
                        path=file.path,
                        start_line=trimmed.start,
                        end_line=trimmed.end,
                        language=file.language,
                        symbol=trimmed.symbol,
                        text=text,
                    )
                )
        return chunks

    # --- Python structure -------------------------------------------------------------------

    def _python_spans(self, file: SourceFile, lines: list[str]) -> list[_Span] | None:
        # A lone "\r" counts as a line break for Python but not for GitHub's line numbers;
        # in that rare case use the line-based splitter so numbers stay correct.
        if "\r" in file.content.replace("\r\n", ""):
            return None
        try:
            tree = ast.parse(file.content)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            logger.debug("python parse failed, using line chunks", extra={"path": file.path})
            return None
        return self._cover(tree.body, 1, len(lines), prefix="")

    def _cover(self, nodes: list[ast.stmt], start: int, end: int, prefix: str) -> list[_Span]:
        """Spans covering lines start..end: one per def/class, plus the code in between."""
        spans: list[_Span] = []
        cursor = start
        for node in nodes:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            node_start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            node_end = node.end_lineno or node.lineno
            if node_start < cursor or node_end > end:
                continue  # defensive: never produce overlapping spans
            if node_start > cursor:
                spans.append(_Span(cursor, node_start - 1))
            spans.extend(self._node_spans(node, node_start, node_end, prefix))
            cursor = node_end + 1
        if cursor <= end:
            spans.append(_Span(cursor, end))
        return spans

    def _node_spans(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        start: int,
        end: int,
        prefix: str,
    ) -> list[_Span]:
        kind = "class" if isinstance(node, ast.ClassDef) else "def"
        name = f"{prefix}{node.name}"
        symbol = f"{kind} {name}"
        if not isinstance(node, ast.ClassDef) or end - start + 1 <= self.max_lines:
            return [_Span(start, end, symbol)]
        # Large class: split into its header, each method, and the code between methods.
        inner = self._cover(node.body, start, end, prefix=f"{name}.")
        return [
            span if span.symbol is not None else _Span(span.start, span.end, symbol)
            for span in inner
        ]

    # --- Size limits ------------------------------------------------------------------------

    def _split_to_limits(self, lines: list[str], span: _Span) -> list[_Span]:
        """Split a span so each piece respects max_lines and max_chars, preferring blank lines."""
        pieces: list[_Span] = []
        start = span.start
        while start <= span.end:
            end = start
            size = len(lines[start - 1]) + 1
            last_blank: int | None = None
            while end < span.end:
                next_size = len(lines[end]) + 1
                if end - start + 1 >= self.max_lines or size + next_size > self.max_chars:
                    break
                end += 1
                size += next_size
                if not lines[end - 1].strip():
                    last_blank = end
            if end < span.end and last_blank is not None and last_blank > (start + end) // 2:
                end = last_blank  # break after a nearby blank line instead of mid-block
            pieces.append(_Span(start, end, span.symbol))
            start = end + 1
        return pieces


def _trim_blank_edges(lines: list[str], span: _Span) -> _Span | None:
    start, end = span.start, span.end
    while start <= end and not lines[start - 1].strip():
        start += 1
    while end >= start and not lines[end - 1].strip():
        end -= 1
    if start > end:
        return None
    return _Span(start, end, span.symbol)


def _chunk_id(repo: RepoRef, commit_sha: str, path: str, span: _Span) -> str:
    key = f"{repo.full_name}@{commit_sha}:{path}:{span.start}-{span.end}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]
