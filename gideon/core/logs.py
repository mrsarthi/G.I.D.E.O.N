"""
G.I.D.E.O.N Logging

One root handler for the whole process, plus the stream binding that console-
less launchers (pythonw) need before anything logs.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from gideon.core.config import LOG_DIR, LOG_LEVEL

_configured = False


def _is_real(stream: object) -> bool:
    """A stream that goes somewhere. pywebview, on import, replaces a None
    stdout/stderr with a writer to `nul`; that is still a process with no
    console, and everything written to it is lost."""
    if stream is None:
        return False
    return str(getattr(stream, "name", "")).lower() not in ("nul", os.devnull.lower())


def bind_headless_streams(name: str, max_bytes: int = 5 * 1024 * 1024) -> Path | None:
    """Give a console-less process real stdout/stderr, returning the log path.

    pythonw.exe has no console, so sys.stdout and sys.stderr are None. Two
    things follow, both silent:

      * logging quietly discards every record — StreamHandler.emit hits
        AttributeError on None and handleError cannot even report it, because
        reporting also goes to sys.stderr. You get no diagnostics at all.
      * anything that writes to stderr eagerly (uvicorn does, on its first
        line) raises and can take the process down before it does any work.

    Must be called before the first get_logger(), because logging captures
    sys.stderr when the handler is constructed — rebinding afterwards leaves
    the handler pointed at None.

    Returns None when real streams already exist, so running from a console
    behaves exactly as before.
    """
    if _is_real(sys.stdout) and _is_real(sys.stderr):
        return None

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{name}.log"

    # Rotated by size, not date: this is a log you read after something went
    # wrong at boot, not an audit trail.
    try:
        if path.exists() and path.stat().st_size > max_bytes:
            path.replace(path.with_suffix(".log.1"))
    except OSError:
        pass

    # Deliberately left open: it backs stdout/stderr for the process lifetime.
    handle = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    sys.stdout = handle
    sys.stderr = handle
    # A handler created before this call captured the old (dead) stream.
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ) and not _is_real(handler.stream):
            handler.setStream(handle)
    return path


def setup_logging() -> None:
    """Install a single root handler. Safe to call more than once."""
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
