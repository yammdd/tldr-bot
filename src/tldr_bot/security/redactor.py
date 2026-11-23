"""Secret redaction utilities to prevent API keys from leaking into logs or Discord error messages."""

from __future__ import annotations

import logging
import re
from typing import Iterable
from urllib.parse import quote

# Patterns matching common LLM provider API key formats and Bearer tokens
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"Bearer\s+[A-Za-z0-9._\-+/=]{12,}", re.IGNORECASE),
    re.compile(r"\bsk-(?:proj-|ant-|or-v1-|live-|test-)?[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b"),
    re.compile(r"\bAQ\.[0-9A-Za-z_\-\.]{20,}\b"),
    re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxai-[A-Za-z0-9]{16,}\b"),
    re.compile(r"(api[_-]?key[\"']?\s*[:=]\s*[\"']?)([^\"'\s&,}]{8,})", re.IGNORECASE),
)


def redact_secrets(text: str, extra_secrets: Iterable[str] = ()) -> str:
    """Replace any known secrets or API-key-like tokens with [REDACTED]."""
    if not text:
        return text

    sanitized = text
    for secret in extra_secrets:
        if not secret or len(secret) < 6:
            continue
        sanitized = sanitized.replace(secret, "[REDACTED]")
        encoded = quote(secret, safe="")
        if encoded != secret:
            sanitized = sanitized.replace(encoded, "[REDACTED]")

    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            sanitized = pattern.sub(r"\1[REDACTED]", sanitized)
        else:
            sanitized = pattern.sub("[REDACTED]", sanitized)

    return sanitized


class SecretRedactingFilter(logging.Filter):
    """Logging filter that strips API-key patterns from log records and formatted exceptions."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_secrets(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: redact_secrets(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    redact_secrets(v) if isinstance(v, str) else v
                    for v in record.args
                )
        return True
