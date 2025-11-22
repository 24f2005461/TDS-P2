from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Generator

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings


@pytest.fixture(autouse=True)
def configure_settings(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """
    Ensure required environment variables exist and cached settings are refreshed.

    This keeps modules that rely on get_settings() from raising ValidationError when the
    test worker environment does not define EMAIL/SECRET/etc.
    """
    monkeypatch.setenv("EMAIL", os.environ.get("EMAIL", "test@example.com"))
    monkeypatch.setenv("SECRET", os.environ.get("SECRET", "super-secret"))
    monkeypatch.setenv(
        "AIPIPE_API_KEY", os.environ.get("AIPIPE_API_KEY", "dummy-token")
    )
    monkeypatch.setenv(
        "AIPIPE_ENDPOINT",
        os.environ.get("AIPIPE_ENDPOINT", "https://aipipe.org/openrouter/v1"),
    )
    monkeypatch.setenv(
        "AIPIPE_GEMINIV1BETA_ENDPOINT",
        os.environ.get(
            "AIPIPE_GEMINIV1BETA_ENDPOINT", "https://aipipe.org/geminiv1beta"
        ),
    )
    monkeypatch.setenv("LLM_DEFAULT_MODEL", "gpt-5-nano")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "x-ai/grok-4.1-fast")

    # Clear cached settings so changes take effect immediately.
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def anyio_backend() -> str:
    """Limit pytest-anyio to asyncio to avoid extra dependencies like trio."""
    return "asyncio"
