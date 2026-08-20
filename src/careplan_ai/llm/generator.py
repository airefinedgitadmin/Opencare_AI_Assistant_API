"""Care plan draft generation via the configured chat model.

The LLM call is the single narrow seam integration tests fake out (see
`tests/integration/test_care_plan_endpoint.py`); everything else in the
request path is exercised for real.
"""

import openai
import tenacity
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from careplan_ai.api.schemas.request import CarePlanGenerationRequest
from careplan_ai.core.config import get_settings
from careplan_ai.core.exceptions import ConfigurationError, InvalidRequestError, LlmGenerationError
from careplan_ai.core.logging import get_logger
from careplan_ai.domain.care_plan import CarePlanDraft
from careplan_ai.llm.client import get_chat_model
from careplan_ai.llm.context_window import fit_to_context_window
from careplan_ai.llm.prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT

logger = get_logger(__name__)


def _has_sufficient_context(payload: CarePlanGenerationRequest) -> bool:
    return bool(
        payload.assessment_responses
        or payload.previous_care_plans
        or payload.review_notes
        or payload.previous_assessment_completions
    )


async def generate_care_plan_draft(payload: CarePlanGenerationRequest) -> CarePlanDraft:
    if not _has_sufficient_context(payload):
        raise InvalidRequestError(
            "At least one of assessmentResponses, previousCarePlans, reviewNotes, "
            "or previousAssessmentCompletions is required"
        )

    settings = get_settings()
    context_fit = fit_to_context_window(payload, settings.max_context_tokens)
    structured_model = get_chat_model().with_structured_output(CarePlanDraft, include_raw=True)
    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    for example_prompt, example_response in FEW_SHOT_EXAMPLES:
        messages.append(HumanMessage(content=example_prompt))
        messages.append(AIMessage(content=example_response))
    messages.append(HumanMessage(content=context_fit.prompt))

    retryer = tenacity.AsyncRetrying(
        stop=tenacity.stop_after_attempt(settings.llm_max_retries),
        wait=tenacity.wait_exponential(multiplier=0.5, max=8),
        # A missing/deprecated deployment is a permanent misconfiguration, not a
        # transient failure - retrying it just adds latency before the same
        # inevitable error, so it's excluded and fails on the first attempt.
        retry=tenacity.retry_if_not_exception_type(openai.NotFoundError),
        reraise=True,
    )
    try:
        result: dict[str, object] = await retryer(structured_model.ainvoke, messages)
    except openai.NotFoundError as exc:
        # Azure returns this when CAREPLAN_AI_AZURE_OPENAI_DEPLOYMENT doesn't exist -
        # e.g. it was renamed or retired. Logged/classified distinctly from a
        # transient LLM failure so this doesn't get silently lumped in with
        # "AI is busy" (Java's circuit breaker treats both as "unavailable," but an
        # operator scanning these logs needs to be able to tell them apart).
        logger.error(
            "llm_deployment_not_found",
            deployment=settings.azure_openai_deployment,
            model_id=settings.model_id,
            error=str(exc),
        )
        raise ConfigurationError(
            "The configured Azure OpenAI deployment was not found - it may have "
            "been renamed, retired, or deprecated"
        ) from exc
    except Exception as exc:
        logger.error("llm_invocation_failed", error=str(exc))
        raise LlmGenerationError("Failed to generate care plan from the language model") from exc

    parsed = result.get("parsed") if isinstance(result, dict) else None
    if not isinstance(parsed, CarePlanDraft):
        logger.error("llm_structured_output_missing", raw=str(result)[:500])
        raise LlmGenerationError("Language model did not return a structured care plan")

    logger.info(
        "care_plan_generated",
        assessment_template_user_id=payload.assessment_template_user_id,
        has_assessment_responses=bool(payload.assessment_responses),
        previous_care_plan_count=len(context_fit.payload.previous_care_plans),
        review_note_count=len(context_fit.payload.review_notes),
        previous_completion_count=len(context_fit.payload.previous_assessment_completions),
        **{f"{name}_dropped": count for name, count in context_fit.dropped_counts.items()},
    )
    return parsed
