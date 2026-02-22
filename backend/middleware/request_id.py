"""Request ID middleware for distributed tracing.

Generates a UUID for every incoming request and stores it in
``structlog.contextvars`` so that all log lines emitted during that
request automatically include ``request_id``.

The ID is also returned in the ``X-Request-ID`` response header so
clients (and the frontend) can correlate logs.
"""

import uuid
from typing import Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"

logger = structlog.get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to every HTTP request.

    1. Reads ``X-Request-ID`` from the incoming request (if provided by a
       gateway or test harness), otherwise generates a new UUID-4.
    2. Binds the ID to structlog context vars so all downstream logs
       include it automatically.
    3. Adds ``X-Request-ID`` to the response headers.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER, str(uuid.uuid4()))

        # Bind to structlog context for this request scope
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        # Store on request state so route handlers can access it
        request.state.request_id = request_id

        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )

        response = await call_next(request)

        response.headers[REQUEST_ID_HEADER] = request_id

        logger.info(
            "request_finished",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
        )

        return response
