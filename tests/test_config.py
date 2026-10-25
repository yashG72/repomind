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


def test_retrieval_settings_defaults_and_validation() -> None:
    settings = Settings(_env_file=None)
    assert settings.search_mode == "hybrid"
    assert settings.search_rrf_k == 60
    assert settings.rerank_enabled is False  # off by default; see the README's evaluation
    assert settings.rerank_max_candidates == 20

    for bad in (
        {"search_mode": "keyword"},
        {"search_rrf_k": 0},
        {"rerank_max_candidates": 0},
        {"rerank_max_candidates": 101},
        {"search_keyword_candidates": 201},
    ):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **bad)


def test_answer_settings_defaults_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "none"  # /ask is off until a provider is configured
    assert settings.llm_model == "claude-opus-5-5"
    assert settings.ask_max_context_chars == 24_000

    monkeypatch.setenv("REPOMIND_LLM_MODEL", "")  # as in older .env files
    assert Settings(_env_file=None).llm_model == "claude-opus-5-5"

    for bad in (
        {"llm_provider": "openai"},
        {"llm_timeout_seconds": 0},
        {"llm_effort": "extreme"},
        {"ask_max_context_chunks": 0},
        {"ask_max_context_chars": 10},
        {"ask_default_top_k": 21},
    ):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **bad)
