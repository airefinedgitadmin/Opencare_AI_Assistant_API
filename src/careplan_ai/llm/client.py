"""Azure OpenAI chat model construction.

A single process-wide client is cached via `lru_cache`; tests that change
`CAREPLAN_AI_*` env vars must call `get_chat_model.cache_clear()` (see
`tests/conftest.py`).
"""

from functools import lru_cache

from langchain_openai import AzureChatOpenAI

from careplan_ai.core.config import get_settings


@lru_cache
def get_chat_model() -> AzureChatOpenAI:
    settings = get_settings()
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        azure_deployment=settings.azure_openai_deployment,
        api_version=settings.azure_openai_api_version,
        temperature=settings.llm_temperature,
        timeout=settings.llm_request_timeout_seconds,
    )
