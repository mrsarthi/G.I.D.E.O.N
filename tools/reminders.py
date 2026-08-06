"""
G.I.D.E.O.N Reminders Tool

Parses natural language time expressions and persists reminders. The
reminder daemon in server.py polls for due rows and pushes them onto the
event bus, so setting one actually results in being told about it.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from config import get_logger
from db import get_db

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

    Falls back to this time tomorrow when nothing is recognised, which is
    deliberately conservative — better late than fired immediately.
    """
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

    log.debug("Unrecognised time expression %r; defaulting to +1 day", time_str)
    return (ref + timedelta(days=1)).replace(microsecond=0)


def set_reminder(text: str, time_str: str) -> str:
    """Save a reminder for a future moment."""
    clean_text = (text or "").strip()
    if not clean_text:
        return "I need to know what to remind you about."

    try:
        from orchestrator import get_turn_origin

        channel, chat_id = get_turn_origin()
        remind_dt = parse_natural_time(time_str)
        reminder_id = get_db().add_reminder(
            text=clean_text,
            remind_at=remind_dt.isoformat(timespec="seconds"),
            channel=channel,
            chat_id=chat_id,
        )
    except Exception as exc:
        log.error("Failed to save reminder: %s", exc)
        return f"Failed to set that reminder: {exc}"

    log.info("Reminder #%s set for %s via %s", reminder_id, remind_dt, channel)
    return (
        f"Reminder #{reminder_id} saved for "
        f"{remind_dt.strftime('%A, %B %d at %I:%M %p')}: '{clean_text}'."
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
    for rid, text, remind_at in reminders:
        try:
            stamp = datetime.fromisoformat(remind_at).strftime("%a, %b %d at %I:%M %p")
        except ValueError:
            stamp = remind_at
        lines.append(f"#{rid} — [{stamp}] {text}")
    return "\n".join(lines)
