"""Validation of user-supplied GitHub repository URLs.

Security approach: the user's URL is only *parsed* to extract an owner and a repository
name, each checked against GitHub's naming rules. RepoMind never requests the URL the user
typed. Later phases build request URLs themselves from the validated owner and name, against
a fixed allowlist of GitHub hosts, so a user cannot point the server at an internal address
(server-side request forgery). The pattern also rejects ports, credentials, query strings,
fragments, percent-encoding and look-alike hosts, simply because none of them can match.
"""

import re

from repomind.core.models import RepoRef

# Owner: 1-39 characters, letters/digits/hyphens, not starting or ending with a hyphen.
_OWNER = r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?)"
# Repository: 1-100 characters of letters, digits, '.', '_' or '-'.
_NAME = r"(?P<name>[A-Za-z0-9._-]{1,100})"
_GITHUB_REPO_URL = re.compile(
    rf"https://(?:www\.)?github\.com/{_OWNER}/{_NAME}/?",
    re.ASCII,
)

MAX_URL_LENGTH = 300


class InvalidRepositoryURL(ValueError):
    """Raised when a URL is not a supported public GitHub repository URL."""


def parse_github_repo_url(url: str) -> RepoRef:
    """Parse `https://github.com/<owner>/<repo>` into a RepoRef, or raise InvalidRepositoryURL."""
    if not isinstance(url, str):
        raise InvalidRepositoryURL("URL must be a string")
    candidate = url.strip()
    if not candidate or len(candidate) > MAX_URL_LENGTH:
        raise InvalidRepositoryURL("URL is empty or too long")

    match = _GITHUB_REPO_URL.fullmatch(candidate)
    if match is None:
        raise InvalidRepositoryURL(
            "Only URLs of the form https://github.com/<owner>/<repository> are supported"
        )

    owner = match.group("owner")
    name = match.group("name")
    if name.endswith(".git"):
        name = name[: -len(".git")]
    if name in {"", ".", ".."} or ".." in name:
        raise InvalidRepositoryURL("Invalid repository name")
    return RepoRef(owner=owner, name=name)
