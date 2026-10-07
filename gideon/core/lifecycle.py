"""
G.I.D.E.O.N Engine Lifecycle

Lets the running FastAPI app ask its own uvicorn server to stop, which is what
makes a *graceful* shutdown reachable from outside the process.

Why not a signal: the tray starts the engine with CREATE_NO_WINDOW, which
gives the child its own console. GenerateConsoleCtrlEvent — the only way to
deliver CTRL_BREAK on Windows — reaches processes sharing the *caller's*
console, so a parent that has no console (the tray runs under pythonw) can
never signal it. The two requirements are mutually exclusive.

Falling back to TerminateProcess is not acceptable either: it skips lifespan
teardown, so the Telegram updater never stops polling and the next start
collides with the old getUpdates consumer (409 Conflict), and SQLite's WAL is
left without a checkpoint.

So the channel is an authenticated loopback request instead, and this module
is the small piece of shared state it needs.
"""

from __future__ import annotations

from typing import Any

from gideon.core.logs import get_logger

log = get_logger("gideon.lifecycle")

_server: Any = None


def bind(server: Any) -> None:
    """Register the uvicorn Server instance. Called by run_engine.py."""
    global _server
    _server = server


def is_bound() -> bool:
    """False when started some other way (run.bat, `uvicorn` directly), in
    which case there is nothing to ask and the caller must fall back."""
    return _server is not None


def request_shutdown() -> bool:
    """Ask uvicorn to stop serving and run lifespan teardown.

    Returns False if no server was registered, so the caller knows the
    graceful path is unavailable rather than silently assuming it worked.
    """
    if _server is None:
        log.warning("Shutdown requested but no uvicorn server is bound")
        return False
    log.info("Graceful shutdown requested")
    _server.should_exit = True
    return True
