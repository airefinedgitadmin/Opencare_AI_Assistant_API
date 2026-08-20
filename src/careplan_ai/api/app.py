"""FastAPI application factory.

All expensive/side-effectful startup work (logging config, startup log line)
happens in `lifespan`, not at import time or inside `create_app()` itself, so
`create_app()` stays cheap and safe to call repeatedly in tests.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from slowapi.middleware import SlowAPIMiddleware

from careplan_ai.api.routers.care_plan import router as care_plan_router
from careplan_ai.api.routers.health import router as health_router
from careplan_ai.core.config import get_settings
from careplan_ai.core.correlation import CorrelationIdMiddleware
from careplan_ai.core.exception_handlers import register_exception_handlers
from careplan_ai.core.logging import configure_logging, get_logger
from careplan_ai.core.rate_limit import limiter

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("service_starting", environment=settings.environment)
    yield
    logger.info("service_stopping")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Care Plan AI API",
        description="Drafts care plan entries from resident assessments and care-plan history.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.limiter = limiter
    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(SlowAPIMiddleware)
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(care_plan_router)
    return app
