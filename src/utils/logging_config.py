"""Logging configuration with secret redaction filter."""

from __future__ import annotations

import logging
import os

_SECRET_KEYS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LITELLM_API_KEY",
]


class SecretRedactionFilter(logging.Filter):
    """Redacts API key values from all log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key in _SECRET_KEYS:
            val = os.environ.get(key, "")
            if val and len(val) > 8:
                record.msg = str(record.msg).replace(val, "[REDACTED]")
                record.args = tuple(
                    str(a).replace(val, "[REDACTED]") if isinstance(a, str) else a
                    for a in (record.args or ())
                )
        return True


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the secret redaction filter applied."""
    logger = logging.getLogger(name)
    if not any(isinstance(f, SecretRedactionFilter) for f in logger.filters):
        logger.addFilter(SecretRedactionFilter())
    return logger
