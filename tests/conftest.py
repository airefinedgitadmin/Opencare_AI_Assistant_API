"""Shared test fixtures.

Settings, the chat model client, and the rate limiter are process-lifetime
singletons in application code - they must not leak state between tests, so
this cache-clearing fixture runs around every test.
"""

import pathlib

import pytest

from careplan_ai.core.config import get_settings
from careplan_ai.core.rate_limit import limiter
from careplan_ai.llm.client import get_chat_model


@pytest.fixture
def base_env(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    # Settings.env_file=".env" is resolved relative to the cwd - without this,
    # tests silently pick up whatever real `.env` happens to sit in the repo
    # root instead of the values set below.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAREPLAN_AI_API_KEY", "test-api-key")
    monkeypatch.setenv("CAREPLAN_AI_AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("CAREPLAN_AI_AZURE_OPENAI_API_KEY", "test-azure-key")
    monkeypatch.setenv("CAREPLAN_AI_AZURE_OPENAI_DEPLOYMENT", "test-deployment")
    # test-deployment is an arbitrary alias, not a real MODEL_REGISTRY entry -
    # model_id must be set explicitly since it can't default to the deployment
    # name the way it does in real config (see Settings._resolve_model_id).
    monkeypatch.setenv("CAREPLAN_AI_MODEL_ID", "gpt-5")


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    get_settings.cache_clear()
    get_chat_model.cache_clear()
    limiter.reset()
    yield
    get_settings.cache_clear()
    get_chat_model.cache_clear()
    limiter.reset()
