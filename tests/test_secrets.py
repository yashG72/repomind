from repomind.ingestion.secrets import REDACTED, redact_secrets


def test_tokens_are_redacted_without_changing_line_count() -> None:
    source = "\n".join(
        [
            "AWS_KEY = 'AKIAABCDEFGHIJKLMNOP'",
            "token = 'ghp_" + "a" * 36 + "'",
            "openai = 'sk-proj-" + "b" * 30 + "'",
            "slack = 'xoxb-1234567890-abcdef'",
            "normal = 'hello world'",
        ]
    )

    redacted, count = redact_secrets(source)

    assert count == 4
    assert redacted.count("\n") == source.count("\n")
    assert "AKIAABCDEFGHIJKLMNOP" not in redacted
    assert "ghp_" not in redacted
    assert redacted.splitlines()[0] == f"AWS_KEY = '{REDACTED}'"
    assert redacted.splitlines()[4] == "normal = 'hello world'"


def test_ordinary_code_is_untouched() -> None:
    source = "def skip_sk(x):\n    return x  # task-123\n"

    assert redact_secrets(source) == (source, 0)
