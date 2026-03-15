"""API key authentication dependency for FastAPI."""

from __future__ import annotations

import os

from fastapi import HTTPException, Request

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_API_KEY: str | None = None


def _get_api_key() -> str | None:
    global _API_KEY
    if _API_KEY is None:
        _API_KEY = os.getenv("API_KEY")
        if not _API_KEY:
            logger.warning("API_KEY not set — all requests will be accepted (dev mode)")
    return _API_KEY


async def verify_api_key(request: Request) -> str:
    """Validate X-API-Key header. Raises 401 if invalid."""
    api_key = _get_api_key()

    # Dev mode: no key configured, allow all
    if not api_key:
        return "dev-mode"

    provided = request.headers.get("X-API-Key")
    if not provided or provided != api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return provided


def reset_api_key_cache() -> None:
    """Reset cached key — for testing only."""
    global _API_KEY
    _API_KEY = None
