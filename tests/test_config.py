from pathlib import Path

import pytest
from pydantic import ValidationError

from repomind.config import Settings


def test_defaults_need_no_environment() -> None:
    settings = Settings(_env_file=None)

    assert settings.environment == "development"
    assert settings.llm_api_key is None


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPOMIND_ENVIRONMENT", "production")
    monkeypatch.setenv("REPOMIND_MAX_FILE_SIZE_BYTES", "1234")

    settings = Settings(_env_file=None)

    assert settings.environment == "production"
    assert settings.max_file_size_bytes == 1234


def test_secrets_are_masked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPOMIND_LLM_API_KEY", "sk-very-secret")

    settings = Settings(_env_file=None)

    assert "sk-very-secret" not in repr(settings)
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "sk-very-secret"


def test_invalid_values_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPOMIND_MAX_FILE_SIZE_BYTES", "-1")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_blank_secrets_mean_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPOMIND_LLM_API_KEY", "")

    assert Settings(_env_file=None).llm_api_key is None


def test_env_example_file_is_loadable() -> None:
    settings = Settings(_env_file=Path(__file__).parents[1] / ".env.example")

    assert settings.environment == "development"
    assert settings.llm_api_key is None
