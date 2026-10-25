"""Anthropic (Claude) implementation of the LLMClient protocol. The only module that imports
the Anthropic SDK.

The reply is constrained to the caller's JSON Schema with structured outputs, so it is always
JSON of the right shape unless it is cut off or declined. Requests have a timeout, and the SDK
retries connection errors, timeouts, 429s and 5xx responses up to `max_retries` times with
backoff. Every failure becomes a short LLMError; the prompt, the key and the provider's
response body are never logged or returned. The model gets no tools.

By default a request that the model's safety classifiers decline is re-run server-side on the
fallback model Anthropic recommends for that kind of decline ("fallbacks": "default").
"""

import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import SecretStr

from repomind.config import Settings
from repomind.generation.errors import (
    LLMAuthenticationError,
    LLMError,
    LLMRateLimited,
    LLMRefused,
    LLMTimeout,
    LLMUnavailable,
    MalformedLLMOutput,
)

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLMClient:
    def __init__(
        self,
        api_key: SecretStr,
        model: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        max_output_tokens: int = 16_000,
        effort: str = "medium",
        refusal_fallback: bool = True,
        client_factory: Callable[[], "anthropic.Anthropic"] | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._max_output_tokens = max_output_tokens
        self._effort = effort
        self._refusal_fallback = refusal_fallback
        self._client_factory = client_factory or self._default_client
        self._client: anthropic.Anthropic | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> "AnthropicLLMClient":
        if settings.llm_api_key is None:
            raise ValueError("REPOMIND_LLM_API_KEY is required for the anthropic provider")
        return cls(
            settings.llm_api_key,
            settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            max_output_tokens=settings.llm_max_output_tokens,
            effort=settings.llm_effort,
            refusal_fallback=settings.llm_refusal_fallback,
        )

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self, system_prompt: str, user_prompt: str, response_schema: dict[str, Any]
    ) -> str:
        # Imported here so the API starts fast and tests that never call a provider skip it.
        import anthropic

        client = self._get_client()
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_output_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "output_config": {
                "effort": self._effort,
                "format": {"type": "json_schema", "schema": response_schema},
            },
        }
        start = time.perf_counter()
        try:
            if self._refusal_fallback:
                response = client.beta.messages.create(
                    **request, betas=[FALLBACK_BETA], fallbacks="default"
                )
            else:
                response = client.messages.create(**request)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise self._fail(
                exc, LLMAuthenticationError("The answer provider rejected the configured API key")
            ) from exc
        except anthropic.NotFoundError as exc:
            raise self._fail(
                exc,
                LLMAuthenticationError(
                    f"The answer provider does not offer the configured model '{self._model}'"
                ),
            ) from exc
        except anthropic.RateLimitError as exc:
            raise self._fail(
                exc, LLMRateLimited("The answer provider is rate limiting requests; try later")
            ) from exc
        except anthropic.APITimeoutError as exc:  # before APIConnectionError, its base class
            raise self._fail(
                exc, LLMTimeout("The answer provider did not respond in time")
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise self._fail(
                exc, LLMUnavailable("The answer provider could not be reached")
            ) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise self._fail(
                    exc, LLMUnavailable("The answer provider is temporarily unavailable")
                ) from exc
            raise self._fail(exc, LLMError("The answer provider rejected the request")) from exc

        stop_reason = getattr(response, "stop_reason", None)
        logger.info(
            "answer generated",
            extra={
                "model": getattr(response, "model", self._model),
                "stop_reason": stop_reason,
                "input_tokens": getattr(response.usage, "input_tokens", None),
                "output_tokens": getattr(response.usage, "output_tokens", None),
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            },
        )
        if stop_reason == "refusal":
            raise LLMRefused("The answer provider declined to answer this question")
        if stop_reason == "max_tokens":
            raise MalformedLLMOutput(
                "The model's answer was cut off; try a narrower question or raise "
                "REPOMIND_LLM_MAX_OUTPUT_TOKENS"
            )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        if not text.strip():
            raise MalformedLLMOutput("The model returned an empty answer")
        return text

    def _get_client(self) -> "anthropic.Anthropic":
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = self._client_factory()
        return self._client

    def _default_client(self) -> "anthropic.Anthropic":
        import anthropic

        return anthropic.Anthropic(
            api_key=self._api_key.get_secret_value(),
            timeout=self._timeout_seconds,
            max_retries=self._max_retries,
        )

    def _fail(self, exc: Exception, error: LLMError) -> LLMError:
        # Status and request id only: the exception message can echo the request or response.
        logger.warning(
            "answer provider request failed",
            extra={
                "model": self._model,
                "error_type": type(exc).__name__,
                "status_code": getattr(exc, "status_code", None),
                "provider_request_id": getattr(exc, "request_id", None),
            },
        )
        return error
