import httpx
import pytest
from fastapi.testclient import TestClient

from repomind.api.dependencies import get_ingestion_service
from repomind.config import Settings
from repomind.ingestion.chunking import CodeChunker
from repomind.ingestion.file_filter import DefaultFileFilter
from repomind.ingestion.github_fetcher import GitHubFetcher
from repomind.ingestion.service import IngestionService
from repomind.main import create_app
from tests.github_fake import SHA, FakeGitHub

DEMO_FILES = {
    "src/app.py": b"import os\n\n\ndef main():\n    return os.getcwd()\n",
    "src/config.py": b"API_KEY = 'AKIAABCDEFGHIJKLMNOP'\n",
    "README.md": b"# Demo\n\nHello.\n",
    "node_modules/lib/index.js": b"module.exports = 1\n",
    "assets/logo.png": b"\x89PNG\x00\x00",
    "package-lock.json": b"{}",
    "deploy/key.pem": b"-----BEGIN PRIVATE KEY-----\n",
    "src/__init__.py": b"",
}


def build_client(fake: FakeGitHub, settings: Settings) -> TestClient:
    app = create_app(settings)

    def service() -> IngestionService:
        fetcher = GitHubFetcher(
            settings, DefaultFileFilter(settings.max_file_size_bytes), transport=fake.transport()
        )
        return IngestionService(fetcher, CodeChunker(settings.chunk_max_lines))

    app.dependency_overrides[get_ingestion_service] = service
    return TestClient(app)


@pytest.fixture
def fake() -> FakeGitHub:
    return FakeGitHub(files=dict(DEMO_FILES))


@pytest.fixture
def api(fake: FakeGitHub, settings: Settings) -> TestClient:
    return build_client(fake, settings)


def test_ingest_returns_a_summary(api: TestClient) -> None:
    response = api.post("/ingest", json={"repo_url": "https://github.com/octo/demo"})

    assert response.status_code == 200
    body = response.json()
    assert body["repository"] == "octo/demo"
    assert body["commit_sha"] == SHA
    assert body["files_discovered"] == 8
    assert body["files_processed"] == 3
    assert body["files_skipped"] == 5
    assert body["skipped_by_reason"] == {
        "empty": 1,
        "excluded_directory": 1,
        "lock_file": 1,
        "secret_file": 1,
        "unsupported_type": 1,
    }
    assert body["languages"] == {"python": 2, "markdown": 1}
    assert body["chunks_created"] == 4  # app.py: imports + def main; config.py; README.md
    assert body["secrets_redacted"] == 1
    assert body["skipped_files_truncated"] is False
    assert {"path": "deploy/key.pem", "reason": "secret_file"} in body["skipped_files"]


def test_secret_values_never_appear_in_the_response(api: TestClient) -> None:
    response = api.post("/ingest", json={"repo_url": "https://github.com/octo/demo"})

    assert "AKIAABCDEFGHIJKLMNOP" not in response.text


def test_service_redacts_chunk_text(fake: FakeGitHub, settings: Settings) -> None:
    fetcher = GitHubFetcher(settings, DefaultFileFilter(10_000), transport=fake.transport())
    result = IngestionService(fetcher, CodeChunker()).ingest("https://github.com/octo/demo")

    config_chunk = next(c for c in result.chunks if c.path == "src/config.py")
    assert config_chunk.text == "API_KEY = '[REDACTED]'"
    assert (config_chunk.start_line, config_chunk.end_line) == (1, 1)


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data",
        "https://github.com.evil.com/octo/demo",
        "https://user:pw@github.com/octo/demo",
        "https://github.com/octo/demo/../../other",
        "file:///etc/passwd",
        "https://gitlab.com/octo/demo",
        "javascript:alert(1)",
    ],
)
def test_malicious_urls_are_rejected_before_any_request(
    url: str, fake: FakeGitHub, api: TestClient
) -> None:
    response = api.post("/ingest", json={"repo_url": url})

    assert response.status_code == 422
    assert fake.requests == []


def test_missing_or_oversized_body_is_rejected(api: TestClient) -> None:
    assert api.post("/ingest", json={}).status_code == 422
    assert api.post("/ingest", json={"repo_url": "x" * 1_000}).status_code == 422


def test_not_found_maps_to_404(settings: Settings) -> None:
    fake = FakeGitHub(overrides={"/repos/octo/demo": lambda r: httpx.Response(404)})

    response = build_client(fake, settings).post(
        "/ingest", json={"repo_url": "https://github.com/octo/demo"}
    )

    assert response.status_code == 404


def test_rate_limit_maps_to_429_with_retry_after(settings: Settings) -> None:
    fake = FakeGitHub(
        overrides={"/repos/octo/demo": lambda r: httpx.Response(429, headers={"retry-after": "30"})}
    )

    response = build_client(fake, settings).post(
        "/ingest", json={"repo_url": "https://github.com/octo/demo"}
    )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "30"
    assert "rate limit" in response.json()["detail"]


def test_too_large_maps_to_413(settings: Settings) -> None:
    fake = FakeGitHub(
        metadata={
            "full_name": "octo/demo",
            "default_branch": "main",
            "private": False,
            "size": 10_000_000,
        }
    )

    response = build_client(fake, settings).post(
        "/ingest", json={"repo_url": "https://github.com/octo/demo"}
    )

    assert response.status_code == 413


def test_malformed_response_maps_to_502(settings: Settings) -> None:
    fake = FakeGitHub(archive=b"garbage")

    response = build_client(fake, settings).post(
        "/ingest", json={"repo_url": "https://github.com/octo/demo"}
    )

    assert response.status_code == 502


def test_unavailable_maps_to_503(settings: Settings) -> None:
    fake = FakeGitHub(overrides={"/repos/octo/demo": lambda r: httpx.Response(503)})

    response = build_client(fake, settings).post(
        "/ingest", json={"repo_url": "https://github.com/octo/demo"}
    )

    assert response.status_code == 503


def test_empty_repository_is_a_successful_empty_summary(settings: Settings) -> None:
    fake = FakeGitHub(overrides={"/commits/main": lambda r: httpx.Response(409)})

    response = build_client(fake, settings).post(
        "/ingest", json={"repo_url": "https://github.com/octo/demo"}
    )

    assert response.status_code == 200
    assert response.json()["files_discovered"] == 0
    assert response.json()["chunks_created"] == 0


def test_github_token_never_appears_in_logs(
    fake: FakeGitHub, capsys: pytest.CaptureFixture[str]
) -> None:
    token = "ghp_" + "s" * 36
    settings = Settings(_env_file=None, environment="test", github_token=token)

    build_client(fake, settings).post("/ingest", json={"repo_url": "https://github.com/octo/demo"})

    captured = capsys.readouterr()
    assert token not in captured.out + captured.err
