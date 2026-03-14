"""Input sanitization for text fields before they enter LLM context."""
from __future__ import annotations
import re
import logging
from typing import Final

logger = logging.getLogger(__name__)

INJECTION_PATTERNS: Final[list[str]] = [
    r"ignore\s+(previous|all|prior)\s+instructions",
    r"disregard\s+(your|the|all)?\s*(previous|instructions|rules)",
    r"you\s+are\s+now\s+a",
    r"system\s*prompt",
    r"act\s+as\s+(if\s+you\s+are|a)\s+\w+",
    r"new\s+instructions?\s*:",
    r"<\s*/?system\s*>",
    r"<\s*/?instruction\s*>",
    r"\[\s*INST\s*\]",
    r"###\s*system",
]

_COMPILED: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE | re.MULTILINE) for p in INJECTION_PATTERNS
]


class InjectionDetectedError(ValueError):
    """Raised when an injection pattern is found in external text."""
    def __init__(self, source: str, pattern: str) -> None:
        super().__init__(
            f"Injection pattern detected in '{source}': pattern={pattern!r}"
        )
        self.source = source
        self.pattern = pattern


def sanitize_external_text(text: str, source: str) -> str:
    """Sanitize external text before it enters LLM context.

    Args:
        text: The text to sanitize (DeviceInfo, email domains, etc.)
        source: Field name for audit logging

    Returns:
        The original text if clean.

    Raises:
        InjectionDetectedError: If an injection pattern is found.
    """
    if not text:
        return text

    for compiled in _COMPILED:
        if compiled.search(text):
            logger.warning(
                "injection_attempt_detected",
                extra={"source": source, "pattern": compiled.pattern},
            )
            raise InjectionDetectedError(source, compiled.pattern)

    return text


def sanitize_transaction_text_fields(tx_data: dict) -> dict:
    """Sanitize all text fields in a raw transaction dict.

    Fields checked: DeviceInfo, P_emaildomain, R_emaildomain
    Returns a copy with any injection patterns removed or raises.
    """
    text_fields = ["DeviceInfo", "P_emaildomain", "R_emaildomain"]
    cleaned = dict(tx_data)
    for field in text_fields:
        value = cleaned.get(field)
        if value and isinstance(value, str):
            sanitize_external_text(value, source=field)
    return cleaned
