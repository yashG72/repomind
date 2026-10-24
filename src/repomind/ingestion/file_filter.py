"""Rules for deciding which repository files are indexed.

The rules are deliberately conservative: a file is indexed only if its extension (or exact
file name) is on the supported list below, it is not inside an excluded directory, and its
content looks like human-written UTF-8 text. Everything else is skipped with a reason, so
the ingestion summary can explain exactly what happened.

Checks run in two steps:
1. check_path: by path and size alone, before any bytes are read.
2. check_content: by the file's bytes (binary, encoding, generated, minified, private keys).
"""

import re
from pathlib import PurePosixPath

from repomind.core.models import SkipReason

# Extension -> language name. Only these files are indexed.
LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".m": "objective-c",
    ".dart": "dart",
    ".lua": "lua",
    ".r": "r",
    ".jl": "julia",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".clj": "clojure",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".ps1": "powershell",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".vue": "vue",
    ".svelte": "svelte",
    ".proto": "protobuf",
    ".graphql": "graphql",
    ".tf": "terraform",
    ".md": "markdown",
    ".rst": "restructuredtext",
    ".txt": "text",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".ini": "ini",
    ".cfg": "ini",
    ".xml": "xml",
    ".gradle": "gradle",
}

# Exact file names (no or unusual extension) that are worth indexing.
LANGUAGE_BY_FILENAME: dict[str, str] = {
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "cmakelists.txt": "cmake",
    "gemfile": "ruby",
    "rakefile": "ruby",
    "procfile": "text",
    "license": "text",
    "readme": "text",
}

# Any path containing one of these directory names is skipped: dependencies, build output,
# caches and tool state, none of which describe the project's own implementation.
EXCLUDED_DIRECTORIES: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        "bower_components",
        "jspm_packages",
        "vendor",
        "third_party",
        "Pods",
        "Carthage",
        ".venv",
        "venv",
        "virtualenv",
        "site-packages",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".eggs",
        "dist",
        "build",
        "target",
        "out",
        "obj",
        ".next",
        ".nuxt",
        ".svelte-kit",
        ".gradle",
        ".idea",
        ".vscode",
        "coverage",
        "htmlcov",
        ".terraform",
        ".cache",
    }
)

LOCK_FILES: frozenset[str] = frozenset(
    {
        "package-lock.json",
        "npm-shrinkwrap.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "bun.lockb",
        "poetry.lock",
        "pipfile.lock",
        "uv.lock",
        "pdm.lock",
        "cargo.lock",
        "gemfile.lock",
        "composer.lock",
        "go.sum",
        "packages.lock.json",
        "podfile.lock",
        "mix.lock",
        "pubspec.lock",
    }
)

# Files that commonly hold credentials. They are never read.
SECRET_FILENAMES: frozenset[str] = frozenset(
    {
        ".env",
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".htpasswd",
        ".pgpass",
        "credentials",
        "credentials.json",
        "secrets.json",
        "secrets.yaml",
        "secrets.yml",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "service-account.json",
        "terraform.tfstate",
    }
)
SECRET_EXTENSIONS: frozenset[str] = frozenset(
    {".pem", ".key", ".p12", ".pfx", ".jks", ".keystore", ".kdbx", ".ppk", ".asc", ".gpg"}
)

GENERATED_SUFFIXES: tuple[str, ...] = (
    ".min.js",
    ".min.css",
    ".bundle.js",
    ".map",
    "_pb2.py",
    "_pb2_grpc.py",
    ".pb.go",
    ".pb.cc",
    ".pb.h",
    ".g.dart",
    ".designer.cs",
)

MAX_PATH_LENGTH = 1024
_GENERATED_MARKER = re.compile(
    rb"(?i)(code generated .* do not edit|@generated|auto-?generated file|do not edit.*generated)"
)
_PRIVATE_KEY = re.compile(rb"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----")
# Lines longer than this, or an average line length above the second value, mean minified code.
_MINIFIED_MAX_LINE = 5_000
_MINIFIED_AVG_LINE = 300


def normalize_archive_path(raw: str) -> str | None:
    """Return a safe, relative POSIX path, or None if the path could escape or is malformed.

    Rejects absolute paths, '..' segments, backslashes, NUL and other control characters.
    The result is only ever used as a label and in citations; files are never written to disk.
    """
    if not raw or len(raw) > MAX_PATH_LENGTH:
        return None
    if "\\" in raw or any(ord(ch) < 32 or ord(ch) == 127 for ch in raw):
        return None
    if raw.startswith("/"):
        return None
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    return "/".join(parts)


class DefaultFileFilter:
    """The standard, documented filtering rules. Implements the FileFilter protocol."""

    def __init__(self, max_file_size_bytes: int) -> None:
        self.max_file_size_bytes = max_file_size_bytes

    def language_for(self, path: str) -> str | None:
        name = PurePosixPath(path).name.lower()
        if name in LANGUAGE_BY_FILENAME:
            return LANGUAGE_BY_FILENAME[name]
        if name.startswith("dockerfile"):
            return "dockerfile"
        return LANGUAGE_BY_EXTENSION.get(PurePosixPath(name).suffix)

    def check_path(self, path: str, size_bytes: int) -> SkipReason | None:
        if normalize_archive_path(path) != path:
            return SkipReason.UNSAFE_PATH
        pure = PurePosixPath(path)
        if any(part in EXCLUDED_DIRECTORIES for part in pure.parts[:-1]):
            return SkipReason.EXCLUDED_DIRECTORY

        name = pure.name.lower()
        if self._is_secret_name(name):
            return SkipReason.SECRET_FILE
        if name in LOCK_FILES:
            return SkipReason.LOCK_FILE
        if name.endswith(GENERATED_SUFFIXES):
            return SkipReason.GENERATED
        if self.language_for(path) is None:
            return SkipReason.UNSUPPORTED_TYPE
        if size_bytes > self.max_file_size_bytes:
            return SkipReason.TOO_LARGE
        if size_bytes == 0:
            return SkipReason.EMPTY
        return None

    def check_content(self, path: str, data: bytes) -> SkipReason | None:
        if not data.strip():
            return SkipReason.EMPTY
        if b"\x00" in data[:8192]:
            return SkipReason.BINARY
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            return SkipReason.NOT_UTF8
        if _PRIVATE_KEY.search(data):
            return SkipReason.SECRET_FILE
        head = b"\n".join(data.splitlines()[:10])
        if _GENERATED_MARKER.search(head):
            return SkipReason.GENERATED
        lines = text.split("\n")
        if max(len(line) for line in lines) > _MINIFIED_MAX_LINE or (
            len(text) / len(lines) > _MINIFIED_AVG_LINE
        ):
            return SkipReason.MINIFIED
        return None

    @staticmethod
    def _is_secret_name(name: str) -> bool:
        if name in SECRET_FILENAMES:
            return True
        # .env.local, .env.production, ... but not the harmless template .env.example
        if name.startswith(".env.") and name not in {
            ".env.example",
            ".env.sample",
            ".env.template",
        }:
            return True
        return PurePosixPath(name).suffix in SECRET_EXTENSIONS
