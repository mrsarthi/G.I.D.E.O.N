"""
G.I.D.E.O.N Storage Models

Typed rows for the records that cross module boundaries. Reminders travel
from SQLite through the delivery daemon, the remote mirror and the tools; a
named type there replaces a mix of tuples and dicts whose keys each caller had
to remember.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class Reminder:
    reminder_id: int
    text: str
    remind_at: str
    # Where the reminder was created. chat_id lets delivery reply into the
    # same Telegram conversation rather than broadcasting.
    channel: str = "web"
    chat_id: str | None = None
    # True once the always-on node has accepted it; see reminders/remote.py.
    remote_synced: bool = False

    @property
    def due(self) -> datetime | None:
        """remind_at parsed, or None when the stored value is malformed."""
        try:
            return datetime.fromisoformat(self.remind_at)
        except (TypeError, ValueError):
            return None


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str
