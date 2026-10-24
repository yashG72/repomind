"""An in-memory stand-in for the GitHub API, used with httpx.MockTransport.

No test touches the real network: every request is answered by FakeGitHub.handle.
"""

import io
import json
import tarfile
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

SHA = "0123456789abcdef0123456789abcdef01234567"
ROOT = "octo-demo-0123456"


def make_tarball(
    files: dict[str, bytes],
    root: str = ROOT,
    customize: Callable[[tarfile.TarFile], None] | None = None,
) -> bytes:
    """Build a GitHub-style tar.gz: every path sits under one root folder."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        root_info = tarfile.TarInfo(root)
        root_info.type = tarfile.DIRTYPE
        tar.addfile(root_info)
        for path, data in files.items():
            info = tarfile.TarInfo(f"{root}/{path}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if customize is not None:
            customize(tar)
    return buffer.getvalue()


@dataclass
class FakeGitHub:
    owner: str = "octo"
    repo: str = "demo"
    files: dict[str, bytes] = field(default_factory=dict)
    archive: bytes | None = None
    metadata: dict | None = None
    sha_body: str = SHA
    # Override the response for a request whose path ends with the key.
    overrides: dict[str, Callable[[httpx.Request], httpx.Response]] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        for suffix, responder in self.overrides.items():
            if path.endswith(suffix) or str(request.url) == suffix:
                return responder(request)

        base = f"/repos/{self.owner}/{self.repo}"
        host = request.url.host
        if host == "api.github.com" and path == base:
            meta = self.metadata or {
                "full_name": f"{self.owner}/{self.repo}",
                "default_branch": "main",
                "private": False,
                "size": 10,
            }
            return httpx.Response(200, content=json.dumps(meta))
        if host == "api.github.com" and path == f"{base}/commits/main":
            return httpx.Response(200, text=self.sha_body)
        if host == "api.github.com" and path == f"{base}/tarball/{SHA}":
            return httpx.Response(
                302,
                headers={
                    "location": f"https://codeload.github.com/{self.owner}/{self.repo}"
                    f"/legacy.tar.gz/{SHA}"
                },
            )
        if host == "codeload.github.com":
            body = self.archive if self.archive is not None else make_tarball(self.files)
            return httpx.Response(200, content=body)
        return httpx.Response(404, json={"message": "Not Found"})
