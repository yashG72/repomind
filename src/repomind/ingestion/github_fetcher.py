"""Fetches a public GitHub repository snapshot as a single tar.gz archive.

Why an archive instead of one request per file: GitHub allows 60 unauthenticated API
requests per hour. Downloading the archive costs three requests (metadata, commit, archive)
no matter how many files the repository has.

How a fetch works:
1. GET /repos/{owner}/{repo}: confirm it exists and is public, read its default branch and
   approximate size, and refuse it early if it is too large.
2. GET /repos/{owner}/{repo}/commits/{branch}: resolve the branch to an exact commit SHA, so
   every citation points to the exact version that was indexed.
3. GET /repos/{owner}/{repo}/tarball/{sha}: GitHub redirects this to codeload.github.com.
   The download is streamed and aborted as soon as it passes max_download_bytes.
4. The archive is read in memory. Files are never written to disk and never executed.
   Only regular files are read; symlinks, hard links and devices are skipped.

Network safety: the only URLs requested are built here from a validated owner and name.
Redirects are followed manually, at most a few times, and only to https URLs on the hosts in
ALLOWED_HOSTS. The optional GitHub token is sent only to api.github.com.
"""

import io
import logging
import re
import tarfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import quote, urljoin, urlsplit

import httpx
from pydantic import BaseModel, ValidationError

from repomind import __version__
from repomind.config import Settings
from repomind.core.interfaces import FileFilter
from repomind.core.models import FetchResult, RepoRef, SkippedFile, SkipReason, SourceFile
from repomind.ingestion.errors import (
    GitHubUnavailable,
    IngestionError,
    MalformedResponse,
    RateLimited,
    RepositoryNotFound,
    RepositoryTooLarge,
    UnsafeRedirect,
)
from repomind.ingestion.file_filter import normalize_archive_path

logger = logging.getLogger(__name__)

API_BASE = "https://api.github.com"
API_HOST = "api.github.com"
ALLOWED_HOSTS: frozenset[str] = frozenset({API_HOST, "codeload.github.com"})
MAX_REDIRECTS = 3
# Upper bound on archive entries examined, as protection against archives with huge numbers
# of tiny entries.
MAX_ARCHIVE_ENTRIES = 200_000
_SHA = re.compile(r"[0-9a-f]{40}")


class _RepoMetadata(BaseModel):
    full_name: str
    default_branch: str
    private: bool
    size: int  # kilobytes, as reported by GitHub (approximate)


class GitHubFetcher:
    """Implements the RepositoryFetcher protocol for public github.com repositories."""

    def __init__(
        self,
        settings: Settings,
        file_filter: FileFilter,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._filter = file_filter
        # Tests inject httpx.MockTransport here; production uses the default network transport.
        self._transport = transport

    def fetch(self, repo: RepoRef) -> FetchResult:
        with self._client() as client:
            meta = self._get_metadata(client, repo)
            sha = self._get_commit_sha(client, repo, meta.default_branch)
            if sha is None:
                logger.info("repository is empty", extra={"repo": repo.full_name})
                return FetchResult(
                    repo=repo,
                    commit_sha="",
                    default_branch=meta.default_branch,
                    files=[],
                    skipped=[],
                    files_discovered=0,
                    bytes_downloaded=0,
                )
            archive = self._download_archive(client, repo, sha)

        result = self._read_archive(repo, sha, meta.default_branch, archive)
        logger.info(
            "repository fetched",
            extra={
                "repo": repo.full_name,
                "commit_sha": sha,
                "bytes_downloaded": result.bytes_downloaded,
                "files_discovered": result.files_discovered,
                "files_kept": len(result.files),
            },
        )
        return result

    # --- HTTP -------------------------------------------------------------------------------

    @contextmanager
    def _client(self) -> Iterator[httpx.Client]:
        with httpx.Client(
            transport=self._transport,
            timeout=self._settings.github_timeout_seconds,
            follow_redirects=False,  # redirects are validated and followed in _send
            headers={"User-Agent": f"RepoMind/{__version__}"},
        ) as client:
            yield client

    def _send(
        self, client: httpx.Client, url: str, accept: str, stream: bool = False
    ) -> httpx.Response:
        """Send a GET, following only allowlisted redirects. The caller closes the response."""
        for _ in range(MAX_REDIRECTS + 1):
            host = _validate_destination(url)
            headers = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28"}
            token = self._settings.github_token
            if token is not None and host == API_HOST:
                headers["Authorization"] = f"Bearer {token.get_secret_value()}"
            try:
                response = client.send(
                    client.build_request("GET", url, headers=headers), stream=stream
                )
            except httpx.TimeoutException as exc:
                raise GitHubUnavailable("GitHub did not respond in time") from exc
            except httpx.TransportError as exc:
                raise GitHubUnavailable("Could not connect to GitHub") from exc

            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("location")
                response.close()
                if not location:
                    raise MalformedResponse("GitHub sent a redirect without a destination")
                url = urljoin(url, location)
                continue
            return response
        raise UnsafeRedirect("Too many redirects from GitHub")

    def _get_metadata(self, client: httpx.Client, repo: RepoRef) -> _RepoMetadata:
        response = self._send(client, _repo_api_url(repo), "application/vnd.github+json")
        _raise_for_status(response, repo)
        try:
            meta = _RepoMetadata.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise MalformedResponse("GitHub returned unexpected repository metadata") from exc

        if meta.private:
            # Only reachable with a token that can see private repositories. The MVP is
            # restricted to public repositories, so treat it as not found.
            raise RepositoryNotFound("Repository not found or not public")
        if meta.size * 1024 > self._settings.max_repo_size_bytes:
            raise RepositoryTooLarge(
                f"Repository is about {meta.size // 1024} MB, above the configured limit"
            )
        return meta

    def _get_commit_sha(self, client: httpx.Client, repo: RepoRef, branch: str) -> str | None:
        """Resolve the branch to a commit SHA, or None if the repository is empty."""
        url = f"{_repo_api_url(repo)}/commits/{quote(branch, safe='')}"
        response = self._send(client, url, "application/vnd.github.sha")
        if response.status_code == 409:  # GitHub: "Git Repository is empty."
            return None
        _raise_for_status(response, repo)
        sha = response.text.strip()
        if not _SHA.fullmatch(sha):
            raise MalformedResponse("GitHub returned an invalid commit SHA")
        return sha

    def _download_archive(self, client: httpx.Client, repo: RepoRef, sha: str) -> bytes:
        limit = self._settings.max_download_bytes
        url = f"{_repo_api_url(repo)}/tarball/{sha}"
        response = self._send(client, url, "application/vnd.github+json", stream=True)
        try:
            _raise_for_status(response, repo)
            declared = response.headers.get("content-length")
            if declared is not None and declared.isdigit() and int(declared) > limit:
                raise RepositoryTooLarge("Repository archive is above the download limit")
            buffer = bytearray()
            try:
                for piece in response.iter_bytes():
                    buffer.extend(piece)
                    if len(buffer) > limit:
                        raise RepositoryTooLarge("Repository archive is above the download limit")
            except httpx.TimeoutException as exc:
                raise GitHubUnavailable("Archive download timed out") from exc
            except httpx.TransportError as exc:
                raise GitHubUnavailable("Archive download was interrupted") from exc
            return bytes(buffer)
        finally:
            response.close()

    # --- Archive ----------------------------------------------------------------------------

    def _read_archive(self, repo: RepoRef, sha: str, branch: str, archive: bytes) -> FetchResult:
        settings = self._settings
        files: list[SourceFile] = []
        skipped: list[SkippedFile] = []
        discovered = 0
        total_size = 0

        try:
            tar = tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz")
        except (tarfile.TarError, OSError, EOFError) as exc:
            raise MalformedResponse("GitHub returned an archive that could not be read") from exc

        root: str | None = None
        try:
            for index, member in enumerate(tar):
                if index >= MAX_ARCHIVE_ENTRIES:
                    raise RepositoryTooLarge("Repository archive has too many entries")
                if root is None:
                    # GitHub archives put everything under one "<owner>-<repo>-<sha>/" folder.
                    root = member.name.split("/", 1)[0]
                if member.isdir():
                    continue
                total_size += max(member.size, 0)
                if total_size > settings.max_repo_size_bytes:
                    raise RepositoryTooLarge("Repository content is above the configured limit")

                discovered += 1
                path = _strip_archive_root(member.name, root)
                if path is None:
                    skipped.append(
                        SkippedFile(path=_label(member.name), reason=SkipReason.UNSAFE_PATH)
                    )
                    continue
                if not member.isfile():
                    skipped.append(SkippedFile(path=path, reason=SkipReason.NOT_REGULAR_FILE))
                    continue
                reason = self._filter.check_path(path, member.size)
                if reason is None and len(files) >= settings.max_files_per_repo:
                    reason = SkipReason.FILE_LIMIT
                if reason is not None:
                    skipped.append(SkippedFile(path=path, reason=reason))
                    continue

                extracted = tar.extractfile(member)
                if extracted is None:
                    skipped.append(SkippedFile(path=path, reason=SkipReason.NOT_REGULAR_FILE))
                    continue
                data = extracted.read(settings.max_file_size_bytes + 1)
                if len(data) > settings.max_file_size_bytes:
                    skipped.append(SkippedFile(path=path, reason=SkipReason.TOO_LARGE))
                    continue
                reason = self._filter.check_content(path, data)
                if reason is not None:
                    skipped.append(SkippedFile(path=path, reason=reason))
                    continue
                files.append(
                    SourceFile(
                        path=path,
                        content=data.decode("utf-8-sig"),
                        size_bytes=len(data),
                        language=self._filter.language_for(path),
                    )
                )
        except IngestionError:
            raise
        except (tarfile.TarError, OSError, EOFError, ValueError) as exc:
            raise MalformedResponse("GitHub returned an archive that could not be read") from exc
        finally:
            tar.close()

        return FetchResult(
            repo=repo,
            commit_sha=sha,
            default_branch=branch,
            files=files,
            skipped=skipped,
            files_discovered=discovered,
            bytes_downloaded=len(archive),
        )


def _repo_api_url(repo: RepoRef) -> str:
    return f"{API_BASE}/repos/{quote(repo.owner, safe='')}/{quote(repo.name, safe='')}"


def _validate_destination(url: str) -> str:
    """Raise UnsafeRedirect unless url is https on an allowlisted GitHub host. Returns the host."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if (
        parts.scheme != "https"
        or host not in ALLOWED_HOSTS
        or parts.username is not None
        or parts.password is not None
        or parts.port not in (None, 443)
    ):
        raise UnsafeRedirect("GitHub redirected to a destination that is not allowed")
    return host


def _raise_for_status(response: httpx.Response, repo: RepoRef) -> None:
    status = response.status_code
    if 200 <= status < 300:
        return
    if status == 429 or (status == 403 and _looks_rate_limited(response)):
        retry_after = _retry_after_seconds(response)
        raise RateLimited(
            "GitHub API rate limit reached. Try again later or configure REPOMIND_GITHUB_TOKEN.",
            retry_after_seconds=retry_after,
        )
    if status in (403, 404, 451):
        raise RepositoryNotFound(f"Repository {repo.full_name} was not found or is not public")
    if status == 401:
        raise GitHubUnavailable("GitHub rejected the configured token")
    if status >= 500:
        raise GitHubUnavailable(f"GitHub returned a server error ({status})")
    raise MalformedResponse(f"GitHub returned an unexpected status ({status})")


def _looks_rate_limited(response: httpx.Response) -> bool:
    return response.headers.get("x-ratelimit-remaining") == "0" or "retry-after" in response.headers


def _retry_after_seconds(response: httpx.Response) -> int | None:
    retry_after = response.headers.get("retry-after", "")
    if retry_after.isdigit():
        return int(retry_after)
    reset = response.headers.get("x-ratelimit-reset", "")
    if reset.isdigit():
        return max(int(reset) - int(time.time()), 0)
    return None


def _strip_archive_root(name: str, root: str) -> str | None:
    """Remove the archive's root folder. Entries outside that folder are unsafe (None)."""
    if not root or not name.startswith(f"{root}/"):
        return None
    return normalize_archive_path(name[len(root) + 1 :])


def _label(raw: str) -> str:
    """A printable, bounded version of an unsafe path, for the skipped-files report."""
    cleaned = "".join(ch if ch.isprintable() else "?" for ch in raw)
    return cleaned[:200]
