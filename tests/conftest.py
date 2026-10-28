"""Shared test fixtures.

Settings, the chat model client, and the rate limiter are process-lifetime
singletons in application code - they must not leak state between tests, so
this cache-clearing fixture runs around every test.
"""

import pathlib

import pytest

from careplan_ai.agents.base.llm import get_agent_model
from careplan_ai.core.config import get_settings
from careplan_ai.core.rate_limit import limiter
from careplan_ai.library.store import get_vector_store
from careplan_ai.llm.client import get_chat_model
from careplan_ai.memory.store import get_session_store


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
def _no_real_agent_model_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Agent tests fake the model calls they care about; any other call fails
    fast here instead of retrying against a real endpoint."""

    async def _refuse(*args: object, **kwargs: object) -> None:
        raise RuntimeError("unexpected real model call in a test")

    monkeypatch.setattr("careplan_ai.agents.care_plan.llm.invoke_structured", _refuse)


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    def clear() -> None:
        get_settings.cache_clear()
        get_chat_model.cache_clear()
        get_agent_model.cache_clear()
        get_session_store.cache_clear()
        get_vector_store.cache_clear()
        limiter.reset()

    clear()
    yield
    clear()
