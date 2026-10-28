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
from careplan_ai.memory.dsn import DbSettings, build_dsn

_DB_FIELDS = set(DbSettings.model_fields)


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

    # --- care plan agent (X-CarePlan-Version: 2) ---
    # Model routing: the primary model runs both agents (care plan, verifier);
    # the optional fallback model, if configured, does the care plan agent's
    # revisions when its first draft fails the checks. Unset primary values fall back to the
    # azure_openai_deployment/model_id pair above; deployment and model id are
    # separate for the same reason they are there.
    agent_primary_deployment: str | None = None
    agent_primary_model_id: str | None = None
    agent_fallback_deployment: str | None = None
    agent_fallback_model_id: str | None = None
    # Only sent to models whose registry entry supports it (GPT-5 family).
    agent_reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = "low"
    # Only used by models that accept temperature (e.g. gpt-4.1-mini).
    agent_temperature: float = 0.0
    agent_redraft_temperature: float = 0.5
    # Reasoning models count their reasoning in this cap, so keep headroom.
    agent_max_response_tokens: int = 8000
    # Evidence is never cut or clipped silently. What the care plan agent reads
    # (estimated tokens): the assessment, previous care plans and notes linked
    # to this pathway always go in (the request is refused if they alone don't
    # fit), then resident-level notes, most pathway-related first, then newest.
    # Anything over budget is listed in the session's `omitted` ids, never lost.
    agent_max_draft_evidence_tokens: int = 60_000
    # The care plan agent's starting pack: must-read records, then for each
    # need (risk answer, previous action, pathway note) its best matches across
    # ALL the resident's records, up to this many tokens. Everything else stays
    # reachable through its search_records tool.
    agent_pack_tokens: int = 12_000
    agent_pack_matches_per_need: int = 3
    # Records returned per search_records call.
    agent_search_results: int = 8
    # Model turns the care plan agent may take (searching + submitting) per run.
    agent_max_turns: int = 6
    # How many times a submitted plan may be sent back (by the code checks or
    # the verifier agent) to be fixed.
    agent_max_revisions: int = 2
    # The verifier agent checks every AI statement against its cited quotes.
    # Off = statements are marked "not verified" (only for when quota is short).
    agent_verify: bool = True
    # On a 429, wait as long as Azure's Retry-After asks, up to this.
    agent_rate_limit_max_wait_seconds: float = 20.0

    # --- care plan style library (OPENCARE-51): RAG for tone/style only ---
    # none = off (drafts use the neutral house style); memory = tests/dev;
    # pinecone = an integrated-embedding Pinecone index, one namespace per home.
    library_provider: Literal["none", "memory", "pinecone"] = "none"
    library_top_k: int = 5
    # Unedited AI-assisted plans are kept but ranked below staff-written and
    # staff-edited ones; set false to leave them out of retrieval altogether.
    library_include_unedited_ai: bool = True
    pinecone_api_key: SecretStr | None = None
    pinecone_index_name: str = "careplan-style-library"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_embed_model: str = "llama-text-embed-v2"

    # --- agent memory: Postgres (local / Supabase / Azure Postgres) ---
    # Either the separate DB_* parts or one MEMORY_DB_URL (which wins if set);
    # see memory/dsn.py. Neither configured = in-memory sessions, which are
    # lost on restart and not shared between replicas - tests/dev only.
    memory_db_url: SecretStr | None = None
    db_host: str | None = None
    db_port: int = 5432
    db_name: str | None = None
    db_user: str = "postgres"
    db_password: SecretStr | None = None
    db_sslmode: str | None = None
    # Drafting sessions (which hold the evidence pack) are deleted this long
    # after they start; saved care plan traces are kept.
    agent_session_ttl_hours: int = 24

    @model_validator(mode="after")
    def _resolve_model_id(self) -> "Settings":
        if self.model_id is None:
            self.model_id = self.azure_openai_deployment
        get_model_spec(self.model_id)  # fail fast at startup on an unregistered model
        return self

    @model_validator(mode="after")
    def _resolve_agent_models(self) -> "Settings":
        if self.agent_primary_deployment is None:
            self.agent_primary_deployment = self.azure_openai_deployment
            if self.agent_primary_model_id is None:
                self.agent_primary_model_id = self.model_id
        elif self.agent_primary_model_id is None:
            self.agent_primary_model_id = self.agent_primary_deployment
        assert self.agent_primary_model_id is not None
        get_model_spec(self.agent_primary_model_id)
        if self.agent_fallback_deployment is not None:
            if self.agent_fallback_model_id is None:
                self.agent_fallback_model_id = self.agent_fallback_deployment
            get_model_spec(self.agent_fallback_model_id)
        return self

    @property
    def memory_db_dsn(self) -> str | None:
        """The usable connection string, or None when the DB isn't configured."""
        return build_dsn(DbSettings.model_validate(self.model_dump(include=_DB_FIELDS)))

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
