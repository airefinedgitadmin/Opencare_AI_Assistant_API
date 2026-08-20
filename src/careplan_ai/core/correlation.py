"""Correlation-ID middleware.

Accepts an inbound `X-Correlation-Id` (so the Java caller can propagate its
own request/audit ID), otherwise mints one. Binds it into structlog's
contextvars for the request lifetime and echoes it back in the response
header, so a single ID ties together the caller's log line, this service's
logs, and any error response's `request_id`.
"""

import contextvars
import uuid
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

CORRELATION_ID_HEADER = "X-Correlation-Id"

_correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


def get_correlation_id() -> str | None:
    return _correlation_id_var.get()


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        correlation_id = request.headers.get(CORRELATION_ID_HEADER) or str(uuid.uuid4())
        token = _correlation_id_var.set(correlation_id)
        structlog.contextvars.bind_contextvars(request_id=correlation_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
            _correlation_id_var.reset(token)
        response.headers[CORRELATION_ID_HEADER] = correlation_id
        return response
