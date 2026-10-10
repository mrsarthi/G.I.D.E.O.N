"""
G.I.D.E.O.N Web API

Assembles the FastAPI app: security middleware, route modules, then the page
and static mounts (last, because "/" matches everything beneath it). The
engine's services come from the composition root and are started and stopped
with the app's lifespan.

Run with:  uvicorn gideon.interfaces.api.app:app   (or run_engine.py)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gideon.app import Services
from gideon.core.config import ALLOWED_HOSTS, CORS_ORIGINS
from gideon.core.logs import get_logger
from gideon.interfaces.api.routes import actions, chat, sessions, system, traces
from gideon.interfaces.api.routes.pages import mount_pages

log = get_logger("gideon.api")


def create_app(services: Services | None = None) -> FastAPI:
    services = services or Services()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await services.start()
        try:
            yield
        finally:
            await services.stop()

    app = FastAPI(title="G.I.D.E.O.N API Server", lifespan=lifespan)
    app.state.services = services

    @app.middleware("http")
    async def validate_host_header(request: Request, call_next):
        """Reject requests whose Host is not a loopback name.

        DNS rebinding works by pointing an attacker-controlled domain at
        127.0.0.1; the browser then treats the attacker page as same-origin,
        so CORS does not help. Checking Host does.
        """
        host = (request.headers.get("host") or "").lower()
        if host and host not in ALLOWED_HOSTS:
            log.warning("Rejected request with unexpected Host header: %r", host)
            return JSONResponse(status_code=400, content={"detail": "Invalid Host header."})
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    app.include_router(system.router)
    app.include_router(sessions.router)
    app.include_router(chat.router)
    app.include_router(actions.router)
    app.include_router(traces.router)
    mount_pages(app)
    return app


app = create_app()
