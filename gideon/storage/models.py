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
from typing import Any


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
    # Set on 'tool_call' and 'tool' rows: which tool, and the call's JSON args.
    tool_name: str | None = None
    tool_args: str | None = None


@dataclass(slots=True)
class PendingAction:
    """A privileged tool call waiting for — or resolved by — the user's approval."""

    action_id: int
    session_id: str
    channel: str
    tool_name: str
    tool_args: dict[str, Any]
    # pending -> running -> approved | failed, or pending -> denied | expired
    status: str
    created_at: str
    result: str | None = None
    resolved_at: str | None = None
    # The agent turn that requested it (traces table), if any.
    trace_id: str | None = None

    def describe(self) -> str:
        """One-line, human-readable form, e.g. `run_terminal_command(command='dir')`."""
        args = ", ".join(f"{k}={v!r}" for k, v in self.tool_args.items() if v not in (None, ""))
        return f"{self.tool_name}({args})"


@dataclass(slots=True)
class Project:
    """A code folder of the user's and its state when last refreshed."""

    path: str
    name: str
    kind: str | None = None          # "Rust", "Python", ...
    description: str | None = None   # first line of the README / manifest
    branch: str | None = None
    dirty: int | None = None         # uncommitted changes; None without git
    last_commit_at: str | None = None
    last_commit: str | None = None
    modified_at: str | None = None
    last_opened_at: str | None = None  # in VS Code
    opened_with: str | None = None     # the app the user last asked for
    summary: str = ""
    refreshed_at: str = ""
    missing: bool = False


@dataclass(slots=True)
class Note:
    note_id: int
    # note | task
    kind: str
    text: str
    done: bool
    created_at: str
