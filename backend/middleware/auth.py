"""Simple API key authentication middleware.

Protects endpoints behind a bearer-token check. The expected key is
read from the ``MARSA_API_KEY`` environment variable. When the variable
is unset (e.g. during local development), authentication is **disabled**
and all requests are allowed through.

Usage in FastAPI::

    from middleware.auth import api_key_auth
    app.add_middleware(APIKeyMiddleware)

Exempt paths (health, docs) are always accessible without a key.
"""

import os
from typing import Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.get_logger(__name__)

# Paths that never require authentication
_EXEMPT_PATHS: set[str] = {
    "/api/health",
    "/docs",
    "/redoc",
    "/openapi.json",
}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validate ``Authorization: Bearer <key>`` against ``MARSA_API_KEY``.

    If ``MARSA_API_KEY`` is empty or unset, the middleware is a no-op
    (development mode).  In production, set the env-var to enable
    enforcement.
    """

    def __init__(self, app, api_key: str | None = None) -> None:  # noqa: ANN001
        super().__init__(app)
        self._api_key = api_key or os.getenv("MARSA_API_KEY", "")
        if self._api_key:
            logger.info("api_key_auth_enabled")
        else:
            logger.info("api_key_auth_disabled", reason="MARSA_API_KEY not set")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip auth when no key is configured (dev mode)
        if not self._api_key:
            return await call_next(request)

        # Skip exempt paths
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        # Allow OPTIONS for CORS preflight
        if request.method == "OPTIONS":
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.warning(
                "auth_missing_bearer",
                path=request.url.path,
                client=request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorized", "message": "Missing or invalid Authorization header"},
            )

        token = auth_header[7:]  # strip "Bearer "
        if token != self._api_key:
            logger.warning(
                "auth_invalid_key",
                path=request.url.path,
                client=request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=403,
                content={"error": "forbidden", "message": "Invalid API key"},
            )

        return await call_next(request)
