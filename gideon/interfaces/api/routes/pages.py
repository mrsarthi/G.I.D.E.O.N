"""HTML pages and static assets.

Only frontend/ and static/ are reachable. The old mount exposed the entire
project directory, gideon.db included.
"""

from __future__ import annotations

from fastapi import APIRouter, FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from gideon.core.config import AUDIO_DIR, FRONTEND_DIR, QUICK_SESSION_ID, STATIC_DIR
from gideon.core.logs import get_logger
from gideon.core.security import API_TOKEN

log = get_logger("gideon.api.pages")

router = APIRouter()


def _inject(html: str, **globals_: str) -> str:
    """Hand values to the page as window globals, inside <head> if present.

    The token reaches the page only through these same-origin routes, so a
    cross-origin attacker cannot read it, and the Host check stops DNS
    rebinding from making itself same-origin.
    """
    assignments = "".join(f'window.{name} = "{value}";' for name, value in globals_.items())
    script = f"<script>{assignments}</script>"
    if "</head>" in html:
        return html.replace("</head>", script + "\n</head>", 1)
    return script + html


@router.get("/", response_class=HTMLResponse)
def serve_index():
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(_inject(html, GIDEON_API_TOKEN=API_TOKEN))


@router.get("/quick", response_class=HTMLResponse)
def serve_quick():
    """The hotkey window's page: one input, one answer, no hologram.

    Registered before the catch-all StaticFiles mount, which would otherwise
    swallow this path. The session id is injected alongside the token so the
    window never has to round-trip through session creation before it can
    accept the first keystroke.
    """
    html = (FRONTEND_DIR / "quick.html").read_text(encoding="utf-8")
    return HTMLResponse(
        _inject(html, GIDEON_API_TOKEN=API_TOKEN, GIDEON_QUICK_SESSION=QUICK_SESSION_ID)
    )


def mount_pages(app: FastAPI) -> None:
    """Register the pages and static mounts. Must run after every API router:
    the "/" StaticFiles mount matches everything beneath it."""
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    if not FRONTEND_DIR.is_dir():  # pragma: no cover - broken checkout
        log.error("Frontend directory %s is missing", FRONTEND_DIR)
        return

    app.include_router(router)
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
