"""Process settings, sourced from `CAREPLAN_AI_*` env vars (or a `.env` file).

`get_settings()` is the only way settings should be obtained - nothing else
in the codebase should read `os.environ` directly. It is cached for the
life of the process; tests that change env vars must call
`get_settings.cache_clear()` (see `tests/conftest.py`).
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from careplan_ai.core.model_config import get_model_spec


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CAREPLAN_AI_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["local", "development", "staging", "production"] = "local"
    log_level: str = "INFO"

    # --- inbound auth: shared secret the Java backend sends as X-Api-Key ---
    api_key: SecretStr

    # --- azure openai ---
    azure_openai_endpoint: str
    azure_openai_api_key: SecretStr
    # The Azure resource's deployment alias - what's actually called over HTTP.
    # This is a human-chosen name, not a reliable indicator of the real model.
    azure_openai_deployment: str
    azure_openai_api_version: str = "2024-08-01-preview"

    # Which entry in core/model_config.py's MODEL_REGISTRY describes the real
    # model behind azure_openai_deployment - deliberately a separate setting
    # from the deployment alias above, since deployment names are arbitrary
    # and can end up pointing at a different model later. Defaults to the
    # deployment name itself, which is correct today (the deployment happens
    # to be named after its model); set this explicitly if that ever diverges.
    # Validated against the registry at startup - see _resolve_model_id below.
    model_id: str | None = None

    # --- llm tuning ---
    llm_temperature: float = 0.2
    llm_max_retries: int = 3
    llm_request_timeout_seconds: float = 30.0

    # --- prompt size safety net: the Java backend caps how much history it
    # fetches per resident, but a resident with unusually long free-text notes
    # or a deep care-plan history could still assemble a prompt too large for
    # the model's context window. Rather than let that fail outright, the
    # oldest previous-care-plans/review-notes are trimmed first until the
    # estimated prompt fits within the limit derived below (see
    # llm/context_window.py). The model's actual context window comes from
    # MODEL_REGISTRY via model_id above - swapping models is a model_id change,
    # not a code change, and this limit moves with it automatically. ---
    max_response_tokens: int = 2000
    # Rough allowance for the system prompt, few-shot examples, and message
    # formatting overhead that isn't the per-request context being sized.
    prompt_overhead_tokens: int = 1500

    # --- rate limiting: protects both this service and the Azure OpenAI
    # deployment's own quota from being overwhelmed by one caller.
    # Scoped per end user (see core/rate_limit.py), not per shared API key, so
    # this is a per-PERSON allowance - one generate call roughly every 6
    # seconds sustained. Tune the actual ceiling against the Azure OpenAI
    # deployment's real TPM/RPM quota; no value set here can exceed what Azure
    # itself allows. ---
    rate_limit_requests: int = 10
    rate_limit_window_seconds: int = 60

    @model_validator(mode="after")
    def _resolve_model_id(self) -> "Settings":
        if self.model_id is None:
            self.model_id = self.azure_openai_deployment
        get_model_spec(self.model_id)  # fail fast at startup on an unregistered model
        return self

    @property
    def max_context_tokens(self) -> int:
        """Derived, not a settable field - stays correct automatically whenever
        model_id points at a different MODEL_REGISTRY entry."""
        assert self.model_id is not None, "_resolve_model_id validator should always set this"
        spec = get_model_spec(self.model_id)
        return spec.context_window_tokens - self.max_response_tokens - self.prompt_overhead_tokens


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # required fields are populated from env
