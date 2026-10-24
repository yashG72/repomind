import pytest
from fastapi.testclient import TestClient

from repomind.config import Settings
from repomind.main import create_app


@pytest.fixture
def settings() -> Settings:
    # _env_file=None: tests must not depend on a developer's local .env file.
    return Settings(_env_file=None, environment="test", log_format="text")


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))
