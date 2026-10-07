"""
G.I.D.E.O.N Turn Context

Which channel and session the current turn belongs to. Tools read this
instead of having it threaded through every signature — set_reminder needs it
so a reminder created on Telegram is delivered back to that chat.

Lives in core so tools can read it without importing the agent.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

current_channel: ContextVar[str] = ContextVar("current_channel", default="web")
current_session: ContextVar[str] = ContextVar("current_session", default="")

TELEGRAM_SESSION_PREFIX = "telegram_"


def telegram_session_id(chat_id: object) -> str:
    return f"{TELEGRAM_SESSION_PREFIX}{chat_id}"


def get_turn_origin() -> tuple[str, str | None]:
    """Return (channel, chat_id) for the turn being processed.

    Telegram session ids are "telegram_<chat_id>", so the chat id is
    recoverable without extra plumbing.
    """
    channel = current_channel.get()
    session = current_session.get()
    chat_id = None
    if channel == "telegram" and session.startswith(TELEGRAM_SESSION_PREFIX):
        chat_id = session.removeprefix(TELEGRAM_SESSION_PREFIX)
    return channel, chat_id


@contextmanager
def turn(channel: str, session_id: str) -> Iterator[None]:
    """Scope a block to one turn's channel and session."""
    token_ch = current_channel.set(channel)
    token_se = current_session.set(session_id)
    try:
        yield
    finally:
        current_channel.reset(token_ch)
        current_session.reset(token_se)
