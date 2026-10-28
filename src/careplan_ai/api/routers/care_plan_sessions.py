"""Care plan agent endpoints (X-CarePlan-Version: 2).

Called only by the Java backend (X-Api-Key), which forwards the frontend's
actions. The model-calling endpoints (start, outcomes, redraft) share the
per-user rate limit with v1's generate.
"""

from fastapi import APIRouter, Depends, Request, status

from careplan_ai.agents.care_plan import service
from careplan_ai.agents.care_plan.schemas import CarePlanTrace, SessionView
from careplan_ai.api.dependencies import require_api_key
from careplan_ai.api.schemas.agent import (
    DecisionRequest,
    FinalizeRequest,
    RedraftRequest,
    StartSessionRequest,
)
from careplan_ai.api.versioning import require_agent_version
from careplan_ai.core.rate_limit import generation_limit_value, limiter

router = APIRouter(
    prefix="/care-plans",
    tags=["care-plan-agent"],
    dependencies=[Depends(require_api_key), Depends(require_agent_version)],
)


@router.post("/sessions", response_model=SessionView, status_code=status.HTTP_201_CREATED)
@limiter.limit(generation_limit_value)
async def start_session(request: Request, payload: StartSessionRequest) -> SessionView:
    return await service.start_session(payload)


@router.get("/sessions/{session_id}", response_model=SessionView)
async def get_session(session_id: str) -> SessionView:
    return await service.get_session(session_id)


@router.post("/sessions/{session_id}/cards/{card_id}/decision", response_model=SessionView)
async def decide(session_id: str, card_id: str, payload: DecisionRequest) -> SessionView:
    return await service.decide(session_id, card_id, payload)


@router.post("/sessions/{session_id}/outcomes", response_model=SessionView)
@limiter.limit(generation_limit_value)
async def write_outcomes(request: Request, session_id: str) -> SessionView:
    """Write outcomes for the actions staff have added (Outcome field opened)."""
    return await service.write_outcomes(session_id)


@router.post("/sessions/{session_id}/cards/{card_id}/redraft", response_model=SessionView)
@limiter.limit(generation_limit_value)
async def redraft(
    request: Request, session_id: str, card_id: str, payload: RedraftRequest
) -> SessionView:
    return await service.redraft(session_id, card_id, payload.target)


@router.post("/sessions/{session_id}/finalize", response_model=CarePlanTrace)
async def finalize(session_id: str, payload: FinalizeRequest) -> CarePlanTrace:
    return await service.finalize(session_id, payload.care_plan_id)


@router.get("/traces/{care_plan_id}", response_model=CarePlanTrace)
async def get_trace(care_plan_id: int) -> CarePlanTrace:
    return await service.get_trace(care_plan_id)
