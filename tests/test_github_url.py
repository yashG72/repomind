import pytest

from repomind.ingestion.github_url import InvalidRepositoryURL, parse_github_repo_url


@pytest.mark.parametrize(
    ("url", "owner", "name"),
    [
        ("https://github.com/tiangolo/fastapi", "tiangolo", "fastapi"),
        ("https://github.com/tiangolo/fastapi/", "tiangolo", "fastapi"),
        ("https://github.com/tiangolo/fastapi.git", "tiangolo", "fastapi"),
        ("https://www.github.com/psf/requests", "psf", "requests"),
        ("  https://github.com/a-b/c.d_e  ", "a-b", "c.d_e"),
    ],
)
def test_valid_urls(url: str, owner: str, name: str) -> None:
    ref = parse_github_repo_url(url)

    assert (ref.owner, ref.name) == (owner, name)
    assert ref.html_url == f"https://github.com/{owner}/{name}"


@pytest.mark.parametrize(
    "url",
    [
        "",
        "github.com/psf/requests",  # no scheme
        "http://github.com/psf/requests",  # plain http
        "ftp://github.com/psf/requests",
        "https://gitlab.com/psf/requests",
        "https://github.com.evil.com/psf/requests",  # look-alike host
        "https://evil.com/github.com/psf/requests",
        "https://user:pass@github.com/psf/requests",  # credentials
        "https://github.com@evil.com/psf/requests",
        "https://github.com:8080/psf/requests",  # port
        "https://127.0.0.1/psf/requests",
        "https://169.254.169.254/latest/meta-data",  # cloud metadata endpoint
        "https://localhost/psf/requests",
        "https://github.com/psf",  # missing repo
        "https://github.com/psf/requests/tree/main",  # extra path
        "https://github.com/psf/requests?ref=x",  # query string
        "https://github.com/psf/requests#readme",  # fragment
        "https://github.com/psf/%2e%2e",  # percent-encoding
        "https://github.com/psf/..",
        "https://github.com/psf/.git",
        "https://github.com/-psf/requests",  # invalid owner
        "https://github.com/psf/req uests",
        "https://github.com/psf/req\nuests",  # embedded newline
        "https://github.com/psf\\requests",
        "https://gіthub.com/psf/requests",  # Cyrillic 'і'
        "https://github.com/" + "a" * 40 + "/repo",  # owner too long
        "https://github.com/psf/" + "r" * 400,  # URL too long
    ],
)
def test_invalid_urls_are_rejected(url: str) -> None:
    with pytest.raises(InvalidRepositoryURL):
        parse_github_repo_url(url)


def test_non_string_is_rejected() -> None:
    with pytest.raises(InvalidRepositoryURL):
        parse_github_repo_url(None)  # type: ignore[arg-type]
