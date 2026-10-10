"""
G.I.D.E.O.N Reminders Tool

Parses natural language time expressions and persists reminders. The
reminder service (reminders/service.py) polls for due rows and delivers them,
so setting one actually results in being told about it.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from gideon.core.context import get_turn_origin
from gideon.core.logs import get_logger
from gideon.storage.db import get_db
from gideon.storage.models import Reminder
from gideon.storage.remote_scheduler import get_remote_scheduler
from gideon.tools.base import Param, tool

log = get_logger("gideon.tools.reminders")

_DAYS_OF_WEEK = [
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
]

_DEFAULT_HOUR = 9  # Unqualified future days land at 9am.


def _apply_clock_time(target: datetime, text: str, default_hour: int = _DEFAULT_HOUR) -> datetime:
    """Overlay an explicit "at 3pm" / "at 15:30" onto a target date."""
    match = re.search(r"(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text)
    if not match:
        match = re.search(r"(?:at\s+)(\d{1,2}):(\d{2})", text)
        if not match:
            return target.replace(hour=default_hour, minute=0, second=0, microsecond=0)
        hour, minute, meridiem = int(match.group(1)), int(match.group(2)), None
    else:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        meridiem = match.group(3)

    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0

    hour = max(0, min(23, hour))
    minute = max(0, min(59, minute))
    return target.replace(hour=hour, minute=minute, second=0, microsecond=0)


def parse_natural_time(time_str: str, now: datetime | None = None) -> datetime:
    """Turn a relative time expression into a concrete datetime.

    Falls back to this time tomorrow when nothing is recognised. Only for
    callers that must produce *a* time; set_reminder uses recognise_time()
    and asks instead (PRD §5.5: never assume).
    """
    ref = now or datetime.now()
    found = recognise_time(time_str, ref)
    if found is not None:
        return found
    log.debug("Unrecognised time expression %r; defaulting to +1 day", time_str)
    return (ref + timedelta(days=1)).replace(microsecond=0)


def recognise_time(time_str: str, now: datetime | None = None) -> datetime | None:
    """The datetime a time expression names, or None if it names none."""
    ref = now or datetime.now()
    clean = (time_str or "").strip().lower()

    # Relative offsets: "in 90 minutes", "in 2 hours", "in 3 weeks"
    offset = re.search(
        r"in\s+(\d+)\s*(minute|min|hour|hr|day|week|month)s?\b", clean
    )
    if offset:
        amount = int(offset.group(1))
        unit = offset.group(2)
        deltas = {
            "minute": timedelta(minutes=amount),
            "min": timedelta(minutes=amount),
            "hour": timedelta(hours=amount),
            "hr": timedelta(hours=amount),
            "day": timedelta(days=amount),
            "week": timedelta(weeks=amount),
            "month": timedelta(days=30 * amount),
        }
        return (ref + deltas[unit]).replace(microsecond=0)

    if "tomorrow" in clean:
        return _apply_clock_time(ref + timedelta(days=1), clean.replace("tomorrow", ""))

    if "tonight" in clean:
        return _apply_clock_time(ref, clean.replace("tonight", ""), default_hour=20)

    if "today" in clean or clean.startswith("at "):
        candidate = _apply_clock_time(ref, clean, default_hour=ref.hour)
        # "at 8am" said at 9am means tomorrow morning.
        if candidate <= ref:
            candidate += timedelta(days=1)
        return candidate

    for idx, day in enumerate(_DAYS_OF_WEEK):
        if day in clean:
            days_ahead = (idx - ref.weekday()) % 7
            if days_ahead == 0 or "next" in clean:
                days_ahead = days_ahead or 7
            return _apply_clock_time(ref + timedelta(days=days_ahead), clean)

    return None


@tool(
    "set_reminder",
    "ONLY call this when the user explicitly instructs you to schedule or save a new future task reminder or alarm.",
    Param(
        "text",
        "The task or note to be reminded about.",
        required=True,
        aliases=("task", "reminder"),
        grounded=True,
        ground_ignore=("remind", "reminder", "set", "me", "to"),
    ),
    Param(
        "time_str",
        "Relative time or date expression (e.g. 'in 5 days', 'tomorrow at 9am', 'in 2 hours').",
        required=True,
        aliases=("time", "when"),
        # No default: a reminder time the user never gave is exactly the
        # invented value PRD §5.5 forbids (it used to be "in 1 day").
        grounded=True,
        ground_ignore=("remind", "reminder"),
    ),
    triggers={"remind", "reminder", "reminders", "alarm", "schedule", "alert"},
)
def set_reminder(text: str, time_str: str) -> str:
    """Save a reminder for a future moment."""
    clean_text = (text or "").strip()
    if not clean_text:
        return "I need to know what to remind you about."

    remind_dt = recognise_time(time_str)
    if remind_dt is None:
        # Never assume (PRD §5.5): an unreadable time is a question, not
        # "tomorrow". The model relays this and asks.
        return (
            f"Not saved: I could not work out when '{time_str}' is. Ask the user for a "
            f"specific time, like 'in 2 hours', 'tomorrow at 9am' or 'Friday at 5pm'."
        )

    try:
        channel, chat_id = get_turn_origin()
        db = get_db()
        reminder_id = db.add_reminder(
            text=clean_text,
            remind_at=remind_dt.isoformat(timespec="seconds"),
            channel=channel,
            chat_id=chat_id,
        )
    except Exception as exc:
        log.error("Failed to save reminder: %s", exc)
        return f"Failed to set that reminder: {exc}"

    # Mirror to the always-on node so it still fires with this machine off.
    # Deliberately after the local commit and deliberately non-fatal: the row
    # exists either way, and remote_synced records who owes the delivery.
    mirrored = _mirror(
        db,
        Reminder(
            reminder_id=reminder_id,
            text=clean_text,
            remind_at=remind_dt.isoformat(timespec="seconds"),
            channel=channel,
            chat_id=chat_id,
        ),
    )

    log.info(
        "Reminder #%s set for %s via %s (remote=%s)",
        reminder_id,
        remind_dt,
        channel,
        "synced" if mirrored else "local-only",
    )
    return (
        f"Reminder #{reminder_id} saved for "
        f"{remind_dt.strftime('%A, %B %d at %I:%M %p')}: '{clean_text}'."
    )


def _mirror(db, reminder: Reminder) -> bool:
    """Best-effort push to the remote scheduler. Never raises."""
    try:
        remote = get_remote_scheduler()
        if not remote.enabled:
            return False
        ok = remote.push(reminder)
        if ok:
            db.mark_reminder_synced(reminder.reminder_id, True)
        return ok
    except Exception as exc:
        log.warning("Remote mirror failed for reminder #%s: %s", reminder.reminder_id, exc)
        return False


@tool(
    "list_reminders",
    "ONLY call this when the user explicitly asks to check, read, or display their currently active saved reminders.",
    triggers={"reminders"},
)
def list_reminders() -> str:
    """List upcoming pending reminders."""
    try:
        reminders = get_db().get_pending_reminders(limit=10)
    except Exception as exc:
        log.error("Failed to fetch reminders: %s", exc)
        return f"Failed to fetch reminders: {exc}"

    if not reminders:
        return "No pending reminders."

    lines = ["Active Reminders:"]
    for reminder in reminders:
        due = reminder.due
        stamp = due.strftime("%a, %b %d at %I:%M %p") if due else reminder.remind_at
        lines.append(f"#{reminder.reminder_id} — [{stamp}] {reminder.text}")
    return "\n".join(lines)
