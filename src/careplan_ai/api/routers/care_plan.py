"""Care plan generation endpoint.

The response field names mirror the Java `care-plan-main-backend`'s
`AssessmentCarePlanOutDTO`/`AssessmentCarePlanResponseDTO` so the caller can
forward the payload (after human review) straight into its own
`POST /care-plan` with no remapping.
"""

from fastapi import APIRouter, Depends, Request, status

from careplan_ai.api.dependencies import require_api_key
from careplan_ai.api.schemas.request import CarePlanGenerationRequest
from careplan_ai.api.schemas.response import CarePlanGenerationResponse
from careplan_ai.core.logging import get_logger
from careplan_ai.core.rate_limit import generation_limit_value, limiter
from careplan_ai.llm.generator import generate_care_plan_draft

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/care-plans", tags=["care-plans"])


@router.post(
    "/generate",
    response_model=CarePlanGenerationResponse,
    status_code=status.HTTP_200_OK,
)
@limiter.limit(generation_limit_value)
async def generate_care_plan(
    request: Request,
    payload: CarePlanGenerationRequest,
    _: None = Depends(require_api_key),
) -> CarePlanGenerationResponse:
    logger.info(
        "care_plan_generation_requested",
        assessment_template_user_id=payload.assessment_template_user_id,
    )
    draft = await generate_care_plan_draft(payload)
    return CarePlanGenerationResponse(
        assessment_template_user_id=payload.assessment_template_user_id,
        assessment_template_id=payload.assessment_template_id,
        action=draft.action,
        outcome=draft.outcome,
        care_plan_title=draft.care_plan_title or payload.care_plan_title,
        rationale=draft.rationale,
    )
