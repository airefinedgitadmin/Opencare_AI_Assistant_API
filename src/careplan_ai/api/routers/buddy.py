"""Open Care Buddy chat endpoint (OPENCARE-102).

Internal only: called by the Java backend with the shared X-Api-Key, never by the browser.
"""

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from careplan_ai.api.dependencies import require_api_key
from careplan_ai.buddy.schemas import BuddyChatRequest, BuddyChatResponse
from careplan_ai.buddy.service import BuddyAiUnavailable, answer
from careplan_ai.core.rate_limit import generation_limit_value, limiter

router = APIRouter(prefix="/buddy", tags=["buddy"])

AI_UNAVAILABLE = {"error": "ai_unavailable"}


@router.post(
    "/chat",
    response_model=BuddyChatResponse,
    status_code=status.HTTP_200_OK,
    responses={502: {"description": "The model failed or took over 30 s"}},
)
@limiter.limit(generation_limit_value)
async def buddy_chat(
    request: Request,
    payload: BuddyChatRequest,
    _: None = Depends(require_api_key),
) -> BuddyChatResponse | JSONResponse:
    try:
        return await answer(payload)
    except BuddyAiUnavailable:
        # Java shows "Open Care Buddy can't answer right now. Please try again."
        return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content=AI_UNAVAILABLE)
