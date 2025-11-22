"""
Utility helpers for configuring and accessing application-wide logging.

The FastAPI app relies on these functions to emit structured logs that
capture request metadata, authentication decisions, and downstream service
calls (e.g., to the quiz scraper or LLM orchestrator). Centralizing logging
setup guarantees consistent formatting and makes it easy to toggle verbosity
based on the environment.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal, Optional

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(request_id)s | %(message)s"
LOG_DATEfmt = "%Y-%m-%d %H:%M:%S"


class RequestContextFilter(logging.Filter):
    """
    Inject a `request_id` attribute so that log records are stable even before
    FastAPI assigns any context. Downstream middleware can update the logger's
    extra dict to propagate real trace IDs.
    """

    def __init__(self, default_request_id: str = "-") -> None:
        super().__init__()
        self._default_request_id = default_request_id

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        if not hasattr(record, "request_id"):
            record.request_id = self._default_request_id
        return True


def configure_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    stream: Optional[logging.Handler] = None,
) -> None:
    """
    Configure the root logger with structured formatting.

    Args:
        level: Minimum severity to emit.
        stream: Optional handler; defaults to stderr if omitted.
    """
    logging.captureWarnings(True)

    handler = stream or logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATEfmt)
    handler.setFormatter(formatter)
    handler.addFilter(RequestContextFilter())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = [handler]


def get_logger(name: str) -> logging.Logger:
    """
    Retrieve a module-specific logger that inherits the global configuration.

    Example:
        logger = get_logger(__name__)
        logger.info("Quiz request received", extra={"request_id": "abc123"})
    """
    return logging.getLogger(name)
