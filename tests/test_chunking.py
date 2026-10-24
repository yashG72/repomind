import textwrap

import pytest

from repomind.core.models import CodeChunk, RepoRef, SourceFile
from repomind.ingestion.chunking import CodeChunker

REPO = RepoRef(owner="octo", name="demo")
SHA = "a" * 40


def make_file(content: str, path: str = "pkg/mod.py", language: str | None = "python"):
    return SourceFile(path=path, content=content, size_bytes=len(content), language=language)


def assert_invariants(content: str, chunks: list[CodeChunk], max_lines: int) -> None:
    """Text matches the original lines, no overlaps, every non-blank line covered once."""
    lines = content.split("\n")
    covered: set[int] = set()
    for chunk in chunks:
        assert chunk.text == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
        assert chunk.end_line - chunk.start_line + 1 <= max_lines
        span = set(range(chunk.start_line, chunk.end_line + 1))
        assert not covered & span, "chunks overlap"
        covered |= span
    non_blank = {i + 1 for i, line in enumerate(lines) if line.strip()}
    assert non_blank <= covered
    assert len({c.chunk_id for c in chunks}) == len(chunks)


PYTHON_SOURCE = textwrap.dedent(
    '''\
    """Module docstring."""
    import os


    def helper(x):
        return x + 1


    @decorator
    @other(1)
    def decorated():
        pass


    class Greeter:
        """Says hello."""

        def greet(self):
            return "hi"


    if __name__ == "__main__":
        helper(1)
    '''
)


def test_python_is_split_by_top_level_definitions() -> None:
    chunks = CodeChunker().chunk(REPO, SHA, make_file(PYTHON_SOURCE))

    summary = [(c.start_line, c.end_line, c.symbol) for c in chunks]
    assert summary == [
        (1, 2, None),
        (5, 6, "def helper"),
        (9, 12, "def decorated"),  # decorators belong to their function
        (15, 19, "class Greeter"),
        (22, 23, None),
    ]
    assert_invariants(PYTHON_SOURCE, chunks, 80)


def test_chunk_metadata_and_github_url() -> None:
    chunk = CodeChunker().chunk(REPO, SHA, make_file(PYTHON_SOURCE))[1]

    assert chunk.path == "pkg/mod.py"
    assert chunk.language == "python"
    assert chunk.commit_sha == SHA
    assert chunk.text == "def helper(x):\n    return x + 1"
    assert chunk.github_url == f"https://github.com/octo/demo/blob/{SHA}/pkg/mod.py#L5-L6"


def test_large_class_is_split_per_method() -> None:
    methods = "\n".join(f"    def m{i}(self):\n        return {i}\n" for i in range(10))
    source = f"class Big:\n    x = 1\n\n{methods}"

    chunks = CodeChunker(max_lines=10).chunk(REPO, SHA, make_file(source))

    symbols = [c.symbol for c in chunks]
    assert symbols[0] == "class Big"  # header with class attributes
    assert symbols[1:] == [f"def Big.m{i}" for i in range(10)]
    assert_invariants(source, chunks, 10)


def test_long_function_is_split_within_limits() -> None:
    body = "\n".join(f"    x{i} = {i}" for i in range(200))
    source = f"def long():\n{body}\n"

    chunks = CodeChunker(max_lines=50).chunk(REPO, SHA, make_file(source))

    assert len(chunks) == 5
    assert all(c.symbol == "def long" for c in chunks)
    assert_invariants(source, chunks, 50)


def test_invalid_python_falls_back_to_line_chunks() -> None:
    source = "def broken(:\n    pass\n" * 30

    chunks = CodeChunker(max_lines=20).chunk(REPO, SHA, make_file(source))

    assert chunks
    assert all(c.symbol is None for c in chunks)
    assert_invariants(source, chunks, 20)


def test_line_chunker_prefers_blank_line_boundaries() -> None:
    block = "\n".join(f"line {i}" for i in range(7))
    source = "\n\n".join([block] * 4)  # four 7-line blocks separated by blank lines

    chunks = CodeChunker(max_lines=10).chunk(REPO, SHA, make_file(source, "notes.md", "markdown"))

    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 7), (9, 15), (17, 23), (25, 31)]
    assert_invariants(source, chunks, 10)


def test_max_chars_limit() -> None:
    source = "\n".join("x" * 99 for _ in range(30))

    chunks = CodeChunker(max_lines=80, max_chars=1_000).chunk(
        REPO, SHA, make_file(source, "a.txt", "text")
    )

    assert all(len(c.text) <= 1_000 for c in chunks)
    assert_invariants(source, chunks, 80)


def test_windows_line_endings_keep_github_line_numbers() -> None:
    source = "import os\r\n\r\ndef f():\r\n    return 1\r\n"

    chunks = CodeChunker().chunk(REPO, SHA, make_file(source))

    assert [(c.start_line, c.end_line, c.symbol) for c in chunks] == [
        (1, 1, None),
        (3, 4, "def f"),
    ]


def test_lone_carriage_return_uses_line_fallback() -> None:
    source = "a = 1\rb = 2\ndef f():\n    pass\n"

    chunks = CodeChunker().chunk(REPO, SHA, make_file(source))

    assert all(c.symbol is None for c in chunks)
    assert_invariants(source, chunks, 80)


@pytest.mark.parametrize("content", ["", "\n\n   \n"])
def test_blank_files_produce_no_chunks(content: str) -> None:
    assert CodeChunker().chunk(REPO, SHA, make_file(content)) == []


def test_python_code_is_never_executed(tmp_path) -> None:
    marker = tmp_path / "pwned"
    source = f"open({str(marker)!r}, 'w').write('x')\nraise SystemExit(1)\n"

    CodeChunker().chunk(REPO, SHA, make_file(source))

    assert not marker.exists()
