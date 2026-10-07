"""Request-scoped dependencies for the API routes."""

from __future__ import annotations

from fastapi import HTTPException, Request

from gideon.app import Services
from gideon.core.logs import get_logger
from gideon.core.security import API_TOKEN_HEADER, token_matches

log = get_logger("gideon.api")


def get_services(request: Request) -> Services:
    return request.app.state.services


def require_api_token(request: Request) -> None:
    """Gate state-changing endpoints behind the shared secret.

    Without this, any process on the machine could POST to 127.0.0.1 and get
    a shell. The token lives in .env and is handed to the frontend by the
    page routes, which inject it into the HTML.
    """
    if not token_matches(request.headers.get(API_TOKEN_HEADER)):
        log.warning(
            "Rejected unauthenticated %s from %s",
            request.url.path,
            request.client.host if request.client else "unknown",
        )
        raise HTTPException(status_code=401, detail="Missing or invalid API token.")
