"""Open Care Buddy chat: one model call per question (OPENCARE-102).

Java owns the data and the safety checks; this only fills the prompt, calls the model and
returns the answer. Nothing from the context is logged: only the request id, the model,
token counts and latency.
"""

import asyncio
import time
from functools import lru_cache
from typing import Any

import openai
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from careplan_ai.buddy.prompt import build_prompt, recent_history, split_answer
from careplan_ai.buddy.schemas import BuddyChatRequest, BuddyChatResponse
from careplan_ai.core.config import get_settings
from careplan_ai.core.logging import get_logger
from careplan_ai.core.model_config import get_model_spec

logger = get_logger(__name__)


class BuddyAiUnavailable(Exception):
    """The model failed or took too long; the caller answers with ai_unavailable."""


@lru_cache
def get_buddy_model() -> AzureChatOpenAI:
    settings = get_settings()
    assert settings.buddy_deployment is not None and settings.buddy_model_id is not None
    spec = get_model_spec(settings.buddy_model_id)
    kwargs: dict[str, Any] = {"max_tokens": settings.buddy_max_response_tokens}
    if spec.supports_temperature:
        kwargs["temperature"] = settings.buddy_temperature
    if spec.supports_reasoning_effort and settings.buddy_reasoning_effort:
        kwargs["reasoning_effort"] = settings.buddy_reasoning_effort
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        azure_deployment=settings.buddy_deployment,
        api_version=settings.azure_openai_api_version,
        model=settings.buddy_model_id,
        timeout=settings.buddy_timeout_seconds,
        # One attempt: the 30 s budget is for the whole answer, and Java shows
        # "try again" on failure rather than keeping the user waiting.
        max_retries=0,
        **kwargs,
    )


def build_messages(request: BuddyChatRequest) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(content=build_prompt(request))]
    for turn in recent_history(request.history):
        messages.append(
            HumanMessage(content=turn.text) if turn.role == "user" else AIMessage(content=turn.text)
        )
    messages.append(HumanMessage(content=request.question))
    return messages


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    # Some models return content parts; keep only the text ones.
    return "".join(
        part.get("text", "") if isinstance(part, dict) else str(part) for part in content
    )


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Azure's own wait hint on a 429 (retry-after-ms / retry-after headers)."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
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


# A retry needs at least this long left in the question's time budget to be worth making.
_RETRY_MIN_LEFT_SECONDS = 5.0


async def _invoke(messages: list[BaseMessage], request_id: str, started: float) -> BaseMessage:
    """The model call, within the question's whole time budget. A 429 (rate limited) is retried
    up to CAREPLAN_AI_BUDDY_RATE_LIMIT_RETRIES times, after Azure's Retry-After (capped), while
    enough of the budget is left. Any other failure is not retried."""
    settings = get_settings()
    attempts = settings.buddy_rate_limit_retries + 1
    for attempt in range(attempts):
        remaining = settings.buddy_timeout_seconds - (time.monotonic() - started)
        try:
            return await asyncio.wait_for(
                get_buddy_model().ainvoke(messages), timeout=max(remaining, 0.1)
            )
        except openai.RateLimitError as exc:
            wait = min(_retry_after_seconds(exc) or 1.0, settings.buddy_rate_limit_max_wait_seconds)
            left = settings.buddy_timeout_seconds - (time.monotonic() - started)
            if attempt + 1 >= attempts or left - wait < _RETRY_MIN_LEFT_SECONDS:
                raise
            logger.warning(
                "buddy_chat_rate_limited",
                request_id=request_id,
                attempt=attempt + 1,
                wait_seconds=round(wait, 2),
            )
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")  # pragma: no cover


async def answer(request: BuddyChatRequest) -> BuddyChatResponse:
    settings = get_settings()
    model_id = settings.buddy_model_id or ""
    started = time.monotonic()
    try:
        result = await _invoke(build_messages(request), request.request_id, started)
    except Exception as exc:
        logger.error(
            "buddy_chat_failed",
            request_id=request.request_id,
            model=model_id,
            latency_ms=int((time.monotonic() - started) * 1000),
            error=type(exc).__name__,
        )
        raise BuddyAiUnavailable from exc
    latency_ms = int((time.monotonic() - started) * 1000)
    answer_markdown, sources = split_answer(_text(result))
    usage = getattr(result, "usage_metadata", None) or {}
    response = BuddyChatResponse(
        answer_markdown=answer_markdown,
        sources=sources,
        model=model_id,
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        latency_ms=latency_ms,
    )
    logger.info(
        "buddy_chat_answered",
        request_id=request.request_id,
        model=model_id,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=latency_ms,
        sources=len(sources),
    )
    return response
