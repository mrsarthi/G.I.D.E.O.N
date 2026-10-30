"""
G.I.D.E.O.N Notes & Tasks

A local scratchpad: notes to look up later and a to-do list. Stored in the
same SQLite database as everything else; nothing leaves the machine.

Kept to three tools on purpose — every schema is prepended to each
tool-enabled prompt, so each one costs context on every turn.
"""

from __future__ import annotations

from gideon.core.logs import get_logger
from gideon.storage.db import get_db
from gideon.tools.base import Param, tool

log = get_logger("gideon.tools.notes")

_KINDS = ("note", "task")


def _kind(value: str | None) -> str | None:
    value = (value or "").strip().lower().rstrip("s")
    if value in ("todo", "to-do", "to do"):
        return "task"
    return value if value in _KINDS else None


@tool(
    "save_note",
    "Save a note or add a task to the user's to-do list. Call when the user asks you to note, jot down, write down or add something to their to-do or task list.",
    Param(
        "text",
        "The note or task, in the user's words.",
        required=True,
        aliases=("note", "task", "content"),
        grounded=True,
        ground_ignore=("note", "task", "todo", "list", "add", "save", "jot"),
    ),
    Param(
        "kind",
        "'task' for a to-do item, 'note' for anything to look up later.",
        enum=_KINDS,
        aliases=("type",),
        default="note",
    ),
    triggers={"note", "notes", "jot", "todo", "todos", "task", "tasks"},
)
def save_note(text: str, kind: str = "note") -> str:
    clean = (text or "").strip()
    if not clean:
        return "There was nothing to save."
    kind = _kind(kind) or "note"
    note_id = get_db().add_note(clean, kind)
    log.info("Saved %s #%s", kind, note_id)
    return f"Saved {kind} #{note_id}: {clean}"


@tool(
    "list_notes",
    "List the user's saved notes or open tasks, optionally only those containing a search word. Call when the user asks what is on their to-do list or what notes they have.",
    Param("kind", "'task' or 'note'. Omit for both.", enum=_KINDS, aliases=("type",), default=None),
    Param(
        "query",
        "Only items containing this word. Omit for all.",
        aliases=("search", "q"),
        default=None,
        grounded=True,
        ground_ignore=("note", "task", "todo"),
    ),
    triggers={"note", "notes", "todo", "todos", "task", "tasks"},
)
def list_notes(kind: str | None = None, query: str | None = None) -> str:
    items = get_db().list_notes(kind=_kind(kind), query=(query or "").strip() or None)
    if not items:
        what = {"task": "open tasks", "note": "notes"}.get(_kind(kind) or "", "notes or open tasks")
        return f"No {what}" + (f" matching '{query}'." if query else ".")
    lines = []
    for item in items:
        label = "Task" if item.kind == "task" else "Note"
        lines.append(f"{label} #{item.note_id}: {item.text}")
    return "\n".join(lines)


@tool(
    "complete_task",
    "Mark a task on the user's to-do list as done, by its number. Call when the user says a task is finished or asks to tick it off.",
    Param(
        "task_id",
        "The task number, as shown by list_notes.",
        type="integer",
        required=True,
        aliases=("id", "number"),
        default=None,
        grounded=True,
    ),
    triggers={"task", "tasks", "todo", "todos"},
)
def complete_task(task_id: int | None) -> str:
    if task_id is None:
        return "Which task? I need its number — list the tasks to see them."
    if get_db().complete_note(task_id):
        return f"Marked task #{task_id} as done."
    return f"There is no open task #{task_id}."
