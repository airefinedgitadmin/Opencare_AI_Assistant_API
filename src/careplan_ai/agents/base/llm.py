"""LLM gateway shared by all agents.

Model routing is by role, resolved from settings:

    primary  - the care plan agent's first draft, and the verifier agent
    fallback - optional stronger model for the care plan agent's revisions
    redraft  - the primary model, with some variation where the model allows it

Temperature and reasoning_effort are only sent to models whose registry entry
supports them (GPT-5-family reasoning models reject custom temperatures).
"""

from functools import lru_cache
from typing import Any, Literal, TypeVar

import openai
import tenacity
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel

from careplan_ai.core.config import get_settings
from careplan_ai.core.exceptions import ConfigurationError, LlmGenerationError
from careplan_ai.core.logging import get_logger
from careplan_ai.core.model_config import get_model_spec

logger = get_logger(__name__)

ModelRole = Literal["primary", "fallback", "redraft"]
T = TypeVar("T", bound=BaseModel)


def model_for_role(role: ModelRole) -> tuple[str, str]:
    """(deployment, model_id) for a role."""
    settings = get_settings()
    if role == "fallback":
        if settings.agent_fallback_deployment is None:
            raise ConfigurationError("No fallback model is configured")
        assert settings.agent_fallback_model_id is not None
        return settings.agent_fallback_deployment, settings.agent_fallback_model_id
    assert settings.agent_primary_deployment is not None
    assert settings.agent_primary_model_id is not None
    return settings.agent_primary_deployment, settings.agent_primary_model_id


def has_fallback_model() -> bool:
    return get_settings().agent_fallback_deployment is not None


@lru_cache
def get_agent_model(role: ModelRole) -> AzureChatOpenAI:
    settings = get_settings()
    deployment, model_id = model_for_role(role)
    spec = get_model_spec(model_id)
    kwargs: dict[str, Any] = {"max_tokens": settings.agent_max_response_tokens}
    if spec.supports_temperature:
        kwargs["temperature"] = (
            settings.agent_redraft_temperature if role == "redraft" else settings.agent_temperature
        )
    if spec.supports_reasoning_effort and settings.agent_reasoning_effort:
        kwargs["reasoning_effort"] = settings.agent_reasoning_effort
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        azure_deployment=deployment,
        api_version=settings.azure_openai_api_version,
        # Lets langchain apply model-specific rules (e.g. GPT-5 temperature).
        model=model_id,
        timeout=settings.llm_request_timeout_seconds,
        max_retries=0,  # retries are handled in invoke_structured, with backoff
        **kwargs,
    )


def _retry_after_seconds(exc: BaseException | None) -> float | None:
    """Azure's own wait hint on a 429 (retry-after-ms / retry-after headers)."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        value = headers.get(name)
        if value:
            try:
                return float(value) * scale
            except ValueError:
                continue
    return None


def _wait(retry_state: tenacity.RetryCallState) -> float:
    """Rate limited: wait as long as Azure asks (capped). Otherwise: short backoff.
    Retrying a 429 after half a second just burns another attempt - the quota
    is per minute."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    backoff: float = tenacity.wait_exponential(multiplier=0.5, max=8)(retry_state)
    if isinstance(exc, openai.RateLimitError):
        hint = _retry_after_seconds(exc)
        cap = get_settings().agent_rate_limit_max_wait_seconds
        return min(hint if hint is not None else max(backoff, 5.0), cap)
    return backoff


def _retryer() -> tenacity.AsyncRetrying:
    return tenacity.AsyncRetrying(
        stop=tenacity.stop_after_attempt(get_settings().llm_max_retries),
        wait=_wait,
        retry=tenacity.retry_if_not_exception_type(
            (openai.NotFoundError, openai.BadRequestError)
        ),
        reraise=True,
    )


async def invoke_with_tools(
    role: ModelRole,
    messages: list[BaseMessage],
    tools: list[type[BaseModel]],
    *,
    force: str | None = None,
) -> AIMessage:
    """One turn of a tool-using agent. The model must call a tool every turn
    (`force` = that specific tool, e.g. the submit tool on the last turn)."""
    deployment, model_id = model_for_role(role)
    bound = get_agent_model(role).bind_tools(tools, tool_choice=force or "required")
    try:
        result: BaseMessage = await _retryer()(bound.ainvoke, messages)
    except openai.NotFoundError as exc:
        logger.error("llm_deployment_not_found", deployment=deployment, model_id=model_id)
        raise ConfigurationError(
            f"Azure OpenAI deployment '{deployment}' was not found - it may have been "
            "renamed, retired, or never created"
        ) from exc
    except Exception as exc:
        logger.error("llm_invocation_failed", role=role, model_id=model_id, error=str(exc))
        raise LlmGenerationError("Failed to get a response from the language model") from exc
    if not isinstance(result, AIMessage):
        raise LlmGenerationError("Language model did not return a message")
    return result


async def invoke_structured(role: ModelRole, schema: type[T], messages: list[BaseMessage]) -> T:
    deployment, model_id = model_for_role(role)
    structured = get_agent_model(role).with_structured_output(schema, include_raw=True)
    try:
        result: dict[str, object] = await _retryer()(structured.ainvoke, messages)
    except openai.NotFoundError as exc:
        logger.error("llm_deployment_not_found", deployment=deployment, model_id=model_id)
        raise ConfigurationError(
            f"Azure OpenAI deployment '{deployment}' was not found - it may have been "
            "renamed, retired, or never created"
        ) from exc
    except Exception as exc:
        logger.error("llm_invocation_failed", role=role, model_id=model_id, error=str(exc))
        raise LlmGenerationError("Failed to get a response from the language model") from exc

    parsed = result.get("parsed") if isinstance(result, dict) else None
    if not isinstance(parsed, schema):
        logger.error("llm_structured_output_missing", role=role, model_id=model_id)
        raise LlmGenerationError("Language model did not return the expected structure")
    return parsed
