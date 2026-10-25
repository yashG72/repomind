"""The Anthropic answer client against a mocked HTTP API: the request it sends and how every
kind of failure becomes a short, safe error. Uses the real SDK; no network, no real key."""

import json
import logging
from collections.abc import Callable

import anthropic
import httpx2
import pytest
from pydantic import SecretStr

from repomind.generation.anthropic_client import FALLBACK_BETA, AnthropicLLMClient
from repomind.generation.errors import (
    LLMAuthenticationError,
    LLMError,
    LLMRateLimited,
    LLMRefused,
    LLMTimeout,
    LLMUnavailable,
    MalformedLLMOutput,
)
from repomind.generation.prompts import RESPONSE_SCHEMA
from repomind.logging_config import configure_logging

KEY = "sk-ant-test-not-a-real-key"
PROMPT = "SECRET-PROMPT-TEXT def private_function(): ..."
Handler = Callable[[httpx2.Request], httpx2.Response]


def message(*blocks: dict, stop_reason: str = "end_turn") -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": list(blocks),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 30},
    }


TEXT = {
    "type": "text",
    "text": '{"answer": "ok [S1]", "citations": ["S1"], "insufficient_evidence": false}',
}


def make_client(handler: Handler, **options) -> tuple[AnthropicLLMClient, list[httpx2.Request]]:
    requests: list[httpx2.Request] = []

    def recording(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return handler(request)

    def factory() -> anthropic.Anthropic:
        return anthropic.Anthropic(
            api_key=KEY,
            base_url="https://api.anthropic.test",
            max_retries=0,
            http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(recording)),
        )

    client = AnthropicLLMClient(
        SecretStr(KEY), "claude-opus-5-5", client_factory=factory, **options
    )
    return client, requests


def respond(status: int, body: dict) -> Handler:
    return lambda request: httpx2.Response(status, json=body)


def error_body(kind: str) -> dict:
    return {"type": "error", "error": {"type": kind, "message": f"{kind}: echo {PROMPT}"}}


def test_request_shape_and_text_reply() -> None:
    thinking = {"type": "thinking", "thinking": "", "signature": "sig"}
    client, requests = make_client(respond(200, message(thinking, TEXT)))

    reply = client.generate("SYSTEM", PROMPT, RESPONSE_SCHEMA)

    assert json.loads(reply)["citations"] == ["S1"]
    [request] = requests
    body = json.loads(request.content)
    assert request.url.path == "/v1/messages"
    assert request.headers["x-api-key"] == KEY
    assert FALLBACK_BETA in request.headers["anthropic-beta"]
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == 16_000
    assert body["system"] == "SYSTEM"
    assert body["messages"] == [{"role": "user", "content": PROMPT}]
    assert body["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
    }
    assert body["fallbacks"] == "default"
    assert "tools" not in body


def test_fallback_can_be_turned_off() -> None:
    client, requests = make_client(respond(200, message(TEXT)), refusal_fallback=False)

    client.generate("SYSTEM", PROMPT, RESPONSE_SCHEMA)

    body = json.loads(requests[0].content)
    assert "fallbacks" not in body
    assert FALLBACK_BETA not in requests[0].headers.get("anthropic-beta", "")


def test_sdk_client_is_created_once_on_first_use() -> None:
    client, requests = make_client(respond(200, message(TEXT)))
    assert client._client is None

    client.generate("S", "U", RESPONSE_SCHEMA)
    first = client._client
    client.generate("S", "U", RESPONSE_SCHEMA)

    assert client._client is first
    assert len(requests) == 2


@pytest.mark.parametrize(
    ("status", "kind", "error"),
    [
        (401, "authentication_error", LLMAuthenticationError),
        (403, "permission_error", LLMAuthenticationError),
        (404, "not_found_error", LLMAuthenticationError),
        (429, "rate_limit_error", LLMRateLimited),
        (500, "api_error", LLMUnavailable),
        (529, "overloaded_error", LLMUnavailable),
        (400, "invalid_request_error", LLMError),
    ],
)
def test_http_errors_become_safe_errors(
    status: int, kind: str, error: type[Exception], caplog: pytest.LogCaptureFixture
) -> None:
    configure_logging("DEBUG", "text")  # the app's logging setup, at its most verbose
    logging.getLogger().addHandler(caplog.handler)  # configure_logging replaced the handlers
    client, _ = make_client(respond(status, error_body(kind)))

    with caplog.at_level(logging.DEBUG), pytest.raises(error) as raised:
        client.generate("SYSTEM", PROMPT, RESPONSE_SCHEMA)

    assert type(raised.value) is error
    message_text = str(raised.value)
    assert KEY not in message_text
    assert "SECRET-PROMPT" not in message_text
    assert KEY not in caplog.text
    assert "SECRET-PROMPT" not in caplog.text
    assert "answer provider request failed" in caplog.text


def test_timeout() -> None:
    def slow(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timed out", request=request)

    client, _ = make_client(slow)

    with pytest.raises(LLMTimeout):
        client.generate("SYSTEM", PROMPT, RESPONSE_SCHEMA)


def test_connection_failure() -> None:
    def down(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    client, _ = make_client(down)

    with pytest.raises(LLMUnavailable):
        client.generate("SYSTEM", PROMPT, RESPONSE_SCHEMA)


def test_refusal() -> None:
    client, _ = make_client(respond(200, message(stop_reason="refusal")))

    with pytest.raises(LLMRefused):
        client.generate("SYSTEM", PROMPT, RESPONSE_SCHEMA)


@pytest.mark.parametrize(
    "reply",
    [message(TEXT, stop_reason="max_tokens"), message(), message({"type": "text", "text": " "})],
    ids=["cut-off", "no-content", "blank"],
)
def test_unusable_replies_are_malformed(reply: dict) -> None:
    client, _ = make_client(respond(200, reply))

    with pytest.raises(MalformedLLMOutput):
        client.generate("SYSTEM", PROMPT, RESPONSE_SCHEMA)


def test_key_is_not_in_repr() -> None:
    client, _ = make_client(respond(200, message(TEXT)))

    assert KEY not in repr(client.__dict__)
