"""Liveness/readiness endpoints. Unversioned since they're infra-facing."""

from fastapi import APIRouter

from careplan_ai.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readiness() -> dict[str, str]:
    get_settings()
    return {"status": "ready"}
