import io
import random
import tarfile

import httpx
import pytest

from repomind.config import Settings
from repomind.core.models import RepoRef, SkipReason
from repomind.ingestion.errors import (
    GitHubUnavailable,
    MalformedResponse,
    RateLimited,
    RepositoryNotFound,
    RepositoryTooLarge,
    UnsafeRedirect,
)
from repomind.ingestion.file_filter import DefaultFileFilter
from repomind.ingestion.github_fetcher import GitHubFetcher
from tests.github_fake import ROOT, SHA, FakeGitHub, make_tarball

REPO = RepoRef(owner="octo", name="demo")


def fetch(fake: FakeGitHub, **overrides):
    settings = Settings(_env_file=None, **overrides)
    fetcher = GitHubFetcher(
        settings, DefaultFileFilter(settings.max_file_size_bytes), transport=fake.transport()
    )
    return fetcher.fetch(REPO)


def respond(status: int, **kwargs):
    return lambda request: httpx.Response(status, **kwargs)


def reasons(result) -> dict[str, SkipReason]:
    return {s.path: s.reason for s in result.skipped}


# --- Happy path -----------------------------------------------------------------------------


def test_fetch_returns_filtered_files_with_metadata() -> None:
    fake = FakeGitHub(
        files={
            "src/app.py": b"print('hi')\n",
            "README.md": b"# Demo\n",
            "node_modules/x/index.js": b"x",
            "logo.png": b"\x89PNG\x00",
            ".env": b"SECRET=1",
        }
    )

    result = fetch(fake)

    assert result.commit_sha == SHA
    assert result.default_branch == "main"
    assert result.files_discovered == 5
    assert {f.path: f.language for f in result.files} == {
        "src/app.py": "python",
        "README.md": "markdown",
    }
    assert result.files[0].content == "print('hi')\n"
    assert reasons(result) == {
        "node_modules/x/index.js": SkipReason.EXCLUDED_DIRECTORY,
        "logo.png": SkipReason.UNSUPPORTED_TYPE,
        ".env": SkipReason.SECRET_FILE,
    }
    assert result.bytes_downloaded > 0


def test_only_allowlisted_github_hosts_are_contacted() -> None:
    fake = FakeGitHub(files={"a.py": b"x = 1\n"})

    fetch(fake)

    assert [r.url.host for r in fake.requests] == [
        "api.github.com",
        "api.github.com",
        "api.github.com",
        "codeload.github.com",
    ]
    assert all(r.url.scheme == "https" for r in fake.requests)


def test_token_is_sent_only_to_the_api_host() -> None:
    fake = FakeGitHub(files={"a.py": b"x = 1\n"})

    fetch(fake, github_token="ghp_" + "t" * 36)

    for request in fake.requests:
        if request.url.host == "api.github.com":
            assert request.headers["authorization"] == "Bearer ghp_" + "t" * 36
        else:
            assert "authorization" not in request.headers


def test_empty_repository_returns_no_files() -> None:
    fake = FakeGitHub(overrides={"/commits/main": respond(409, json={"message": "empty"})})

    result = fetch(fake)

    assert result.files == [] and result.skipped == [] and result.files_discovered == 0


# --- Redirect safety ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "location",
    [
        "https://evil.example.com/archive.tar.gz",
        "http://codeload.github.com/octo/demo/legacy.tar.gz/x",  # downgrade to http
        "https://169.254.169.254/latest/meta-data/",
        "https://localhost/archive",
        "https://user:pw@codeload.github.com/archive",
        "https://codeload.github.com:8443/archive",
        "https://codeload.github.com.evil.com/archive",
        "file:///etc/passwd",
    ],
)
def test_redirects_outside_the_allowlist_are_refused(location: str) -> None:
    fake = FakeGitHub(overrides={f"/tarball/{SHA}": respond(302, headers={"location": location})})

    with pytest.raises(UnsafeRedirect):
        fetch(fake)
    assert all(r.url.host in {"api.github.com", "codeload.github.com"} for r in fake.requests)


def test_redirect_loops_are_bounded() -> None:
    loop = respond(
        302, headers={"location": f"https://api.github.com/repos/octo/demo/tarball/{SHA}"}
    )
    fake = FakeGitHub(overrides={f"/tarball/{SHA}": loop})

    with pytest.raises(UnsafeRedirect):
        fetch(fake)


def test_redirect_without_location_is_malformed() -> None:
    fake = FakeGitHub(overrides={f"/tarball/{SHA}": respond(302)})

    with pytest.raises(MalformedResponse):
        fetch(fake)


def test_renamed_repository_redirect_within_api_is_followed() -> None:
    fake = FakeGitHub(files={"a.py": b"x = 1\n"})
    real_handle = fake.handle
    calls = {"metadata": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/octo/demo":
            calls["metadata"] += 1
            if calls["metadata"] == 1:  # GitHub answers 301 for a renamed repository
                return httpx.Response(301, headers={"location": "/repos/octo/demo"})
        return real_handle(request)

    fake.handle = handle  # type: ignore[method-assign]

    assert fetch(fake).commit_sha == SHA
    assert calls["metadata"] == 2


# --- GitHub API failures --------------------------------------------------------------------


def test_missing_repository() -> None:
    fake = FakeGitHub(overrides={"/repos/octo/demo": respond(404, json={"message": "Not Found"})})

    with pytest.raises(RepositoryNotFound):
        fetch(fake)


def test_private_repository_is_refused() -> None:
    fake = FakeGitHub(
        metadata={"full_name": "octo/demo", "default_branch": "main", "private": True, "size": 1}
    )

    with pytest.raises(RepositoryNotFound):
        fetch(fake)


def test_primary_rate_limit() -> None:
    fake = FakeGitHub(
        overrides={
            "/repos/octo/demo": respond(
                403,
                headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "9999999999"},
                json={"message": "API rate limit exceeded"},
            )
        }
    )

    with pytest.raises(RateLimited) as caught:
        fetch(fake)
    assert caught.value.retry_after_seconds is not None and caught.value.retry_after_seconds > 0


def test_secondary_rate_limit_with_retry_after() -> None:
    fake = FakeGitHub(overrides={"/repos/octo/demo": respond(429, headers={"retry-after": "60"})})

    with pytest.raises(RateLimited) as caught:
        fetch(fake)
    assert caught.value.retry_after_seconds == 60


def test_forbidden_without_rate_limit_is_not_found() -> None:
    fake = FakeGitHub(overrides={"/repos/octo/demo": respond(403, json={"message": "blocked"})})

    with pytest.raises(RepositoryNotFound):
        fetch(fake)


def test_server_error() -> None:
    fake = FakeGitHub(overrides={"/repos/octo/demo": respond(502)})

    with pytest.raises(GitHubUnavailable):
        fetch(fake)


def test_timeout() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(GitHubUnavailable):
        fetch(FakeGitHub(overrides={"/repos/octo/demo": timeout}))


def test_connection_error() -> None:
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(GitHubUnavailable):
        fetch(FakeGitHub(overrides={"/repos/octo/demo": refused}))


@pytest.mark.parametrize(
    "body",
    [b"not json", b"[]", b'{"full_name": "octo/demo"}', b'{"default_branch": 5, "private": "x"}'],
)
def test_malformed_metadata(body: bytes) -> None:
    fake = FakeGitHub(overrides={"/repos/octo/demo": respond(200, content=body)})

    with pytest.raises(MalformedResponse):
        fetch(fake)


@pytest.mark.parametrize("sha", ["", "not-a-sha", "../../etc", "A" * 40, "0" * 39])
def test_invalid_commit_sha(sha: str) -> None:
    with pytest.raises(MalformedResponse):
        fetch(FakeGitHub(sha_body=sha))


def test_corrupt_archive() -> None:
    with pytest.raises(MalformedResponse):
        fetch(FakeGitHub(archive=b"this is not a gzip file"))


def test_truncated_archive() -> None:
    archive = make_tarball({f"f{i}.py": b"x = 1\n" * 500 for i in range(5)})

    with pytest.raises(MalformedResponse):
        fetch(FakeGitHub(archive=archive[: len(archive) // 2]))


# --- Size limits ----------------------------------------------------------------------------


def test_repository_too_large_by_metadata_is_refused_before_download() -> None:
    fake = FakeGitHub(
        metadata={
            "full_name": "octo/demo",
            "default_branch": "main",
            "private": False,
            "size": 500_000,
        }
    )

    with pytest.raises(RepositoryTooLarge):
        fetch(fake, max_repo_size_bytes=100_000_000)
    assert all("tarball" not in r.url.path for r in fake.requests)


def test_download_aborts_past_the_limit() -> None:
    big = make_tarball({"data.txt": random.Random(0).randbytes(50_000)})  # incompressible

    with pytest.raises(RepositoryTooLarge):
        fetch(FakeGitHub(archive=big), max_download_bytes=10_000)


def test_declared_content_length_over_limit() -> None:
    fake = FakeGitHub(
        overrides={f"/legacy.tar.gz/{SHA}": respond(200, headers={"content-length": "999999999"})}
    )

    with pytest.raises(RepositoryTooLarge):
        fetch(fake, max_download_bytes=1_000)


def test_uncompressed_size_limit() -> None:
    archive = make_tarball({f"f{i}.py": b"#" * 5_000 for i in range(10)})

    with pytest.raises(RepositoryTooLarge):
        fetch(FakeGitHub(archive=archive), max_repo_size_bytes=20_000)


def test_oversized_file_is_skipped_not_fatal() -> None:
    fake = FakeGitHub(files={"big.py": b"x = 1\n" * 1_000, "small.py": b"y = 2\n"})

    result = fetch(fake, max_file_size_bytes=1_000)

    assert [f.path for f in result.files] == ["small.py"]
    assert reasons(result) == {"big.py": SkipReason.TOO_LARGE}


def test_file_count_limit() -> None:
    fake = FakeGitHub(files={f"m{i}.py": b"x = 1\n" for i in range(5)})

    result = fetch(fake, max_files_per_repo=3)

    assert len(result.files) == 3
    assert list(reasons(result).values()) == [SkipReason.FILE_LIMIT] * 2


# --- Hostile archive contents ---------------------------------------------------------------


def _add(tar: tarfile.TarFile, name: str, data: bytes = b"x = 1\n", **attrs) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    for key, value in attrs.items():
        setattr(info, key, value)
    tar.addfile(info, io.BytesIO(data) if info.isfile() else None)


def test_traversal_and_absolute_paths_are_skipped() -> None:
    def customize(tar: tarfile.TarFile) -> None:
        _add(tar, f"{ROOT}/../../etc/evil.py")
        _add(tar, f"{ROOT}/src/../../escape.py")
        _add(tar, "/abs/evil.py")
        _add(tar, f"{ROOT}/win\\evil.py")
        _add(tar, f"{ROOT}/ok.py")

    result = fetch(FakeGitHub(archive=make_tarball({}, customize=customize)))

    assert [f.path for f in result.files] == ["ok.py"]
    assert [s.reason for s in result.skipped] == [SkipReason.UNSAFE_PATH] * 4


def test_links_and_special_files_are_not_followed() -> None:
    def customize(tar: tarfile.TarFile) -> None:
        _add(tar, f"{ROOT}/link.py", b"", type=tarfile.SYMTYPE, linkname="/etc/passwd")
        _add(tar, f"{ROOT}/hard.py", b"", type=tarfile.LNKTYPE, linkname=f"{ROOT}/ok.py")
        _add(tar, f"{ROOT}/fifo.py", b"", type=tarfile.FIFOTYPE)
        _add(tar, f"{ROOT}/ok.py")

    result = fetch(FakeGitHub(archive=make_tarball({}, customize=customize)))

    assert [f.path for f in result.files] == ["ok.py"]
    assert reasons(result) == {
        "link.py": SkipReason.NOT_REGULAR_FILE,
        "hard.py": SkipReason.NOT_REGULAR_FILE,
        "fifo.py": SkipReason.NOT_REGULAR_FILE,
    }


def test_nothing_is_written_to_disk(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    fetch(FakeGitHub(files={"a.py": b"x = 1\n", "evil.sh": b"rm -rf /\n"}))

    assert list(tmp_path.iterdir()) == []
