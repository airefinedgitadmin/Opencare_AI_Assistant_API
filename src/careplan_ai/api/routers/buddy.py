"""Open Care Buddy chat endpoint (OPENCARE-102).

Internal only: called by the Java backend with the shared X-Api-Key, never by the browser.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, status
from fastapi.responses import JSONResponse

from careplan_ai.api.dependencies import require_api_key
from careplan_ai.buddy import kb
from careplan_ai.buddy.kb import KbIndexResult
from careplan_ai.buddy.schemas import BuddyChatRequest, BuddyChatResponse
from careplan_ai.buddy.service import BuddyAiUnavailable, answer
from careplan_ai.core.config import get_settings
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


@router.post(
    "/kb/index",
    response_model=KbIndexResult,
    responses={
        413: {"description": "File over the size limit"},
        422: {"description": "No readable text in the file"},
        502: {"description": "The model failed or its suggestion wasn't valid"},
    },
)
@limiter.limit(generation_limit_value)
async def buddy_kb_index(
    request: Request,
    file: Annotated[UploadFile, File()],
    _: Annotated[None, Depends(require_api_key)],
    home_name: Annotated[str, Form()] = "",
    emergency_ids: Annotated[str, Form()] = "",
) -> KbIndexResult | JSONResponse:
    """OPENCARE-109: suggested title, summary, tags, type and main points for a home document.
    Nothing is stored here; Java shows the suggestion for the manager to check and save."""
    settings = get_settings()
    data = await file.read(settings.buddy_kb_max_bytes + 1)
    if len(data) > settings.buddy_kb_max_bytes:
        return JSONResponse(status_code=413, content={"error": "too_large"})
    ids = [
        e.strip() for e in emergency_ids.split(",") if e.strip()
    ] or settings.buddy_kb_emergency_ids
    try:
        return await kb.index(file.filename or "document", data, ids)
    except kb.NoReadableText:
        # Java opens the form empty: "Couldn't read this file. Please fill in the details."
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"error": "no_text"}
        )
    except kb.KbIndexFailed:
        return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content=AI_UNAVAILABLE)
