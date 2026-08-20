"""Structured JSON logging, designed for container log-driver ingestion - no
code change needed between environments.
"""

import logging
import sys

import structlog
from structlog.typing import Processor


def configure_logging(log_level: str = "INFO") -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[*shared_processors, structlog.processors.JSONRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    for noisy_logger in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(noisy_logger).handlers = []
        logging.getLogger(noisy_logger).propagate = False


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
