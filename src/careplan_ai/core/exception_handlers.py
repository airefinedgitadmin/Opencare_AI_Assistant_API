"""Global FastAPI exception handlers.

Exactly two handlers: one for `CarePlanAiError` (and every subclass -
Starlette walks the MRO, so no per-subclass registration is needed) and one
catch-all for anything unexpected, so a bug never leaks a raw traceback to
the caller.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi.errors import RateLimitExceeded

from careplan_ai.core.config import get_settings
from careplan_ai.core.correlation import get_correlation_id
from careplan_ai.core.exceptions import CarePlanAiError
from careplan_ai.core.logging import get_logger

logger = get_logger(__name__)


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: str | None = None


async def _handle_care_plan_ai_error(request: Request, exc: CarePlanAiError) -> JSONResponse:
    logger.error(
        "service_error",
        error_code=exc.error_code,
        path=request.url.path,
        detail=exc.message,
    )
    body = ErrorResponse(
        error_code=exc.error_code, message=exc.message, request_id=get_correlation_id()
    )
    return JSONResponse(status_code=exc.http_status, content=body.model_dump())


async def _handle_rate_limit_exceeded(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    logger.error("rate_limit_exceeded", path=request.url.path, detail=exc.detail)
    body = ErrorResponse(
        error_code="rate_limit_exceeded",
        message=f"Rate limit exceeded: {exc.detail}",
        request_id=get_correlation_id(),
    )
    response = JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS, content=body.model_dump()
    )
    response.headers["Retry-After"] = str(get_settings().rate_limit_window_seconds)
    return response


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_error", path=request.url.path, detail=str(exc), exc_info=True)
    body = ErrorResponse(
        error_code="internal_error",
        message="An unexpected error occurred",
        request_id=get_correlation_id(),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=body.model_dump()
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(CarePlanAiError, _handle_care_plan_ai_error)  # type: ignore[arg-type]
    app.add_exception_handler(RateLimitExceeded, _handle_rate_limit_exceeded)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _handle_unexpected_error)
