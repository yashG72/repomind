"""Redaction of credential-like strings inside otherwise useful source files.

Whole files that are obviously secret (.env, private keys) are skipped by the file filter.
This module handles the other case: a normal source file that happens to contain a token.
Each match is replaced on the same line, so line numbers stay correct for citations.
Redacted values are never logged.
"""

import re

REDACTED = "[REDACTED]"

_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b"),  # GitHub tokens
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,255}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),  # GitLab
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),  # Slack
    re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}\b"),  # OpenAI / Anthropic style
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),  # Google API key
    re.compile(r"\b(?:sk|rk)_live_[0-9A-Za-z]{20,}\b"),  # Stripe
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),  # JWT
)


def redact_secrets(text: str) -> tuple[str, int]:
    """Return (text with credential-like tokens replaced, number of replacements)."""
    total = 0
    for pattern in _TOKEN_PATTERNS:
        text, count = pattern.subn(REDACTED, text)
        total += count
    return text, total
