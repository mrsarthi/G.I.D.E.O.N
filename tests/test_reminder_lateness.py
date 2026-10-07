"""Reminders that came due while the machine was off.

The old daemon selected every pending row with remind_at <= now and fired them
all in one pass, each with its own synthesised speech. Coming back from a
weekend away meant a stampede, with Friday's 9am alert shouting on Monday
evening. These tests pin the grace window and the digest that replaced it.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from gideon.reminders.service import ReminderService, partition_due_reminders
from gideon.storage.db import get_db
from gideon.storage.models import Reminder


def _row(minutes_late: int, reminder_id: int = 1, **extra) -> Reminder:
    fields = {
        "reminder_id": reminder_id,
        "text": f"reminder {reminder_id}",
        "remind_at": (
            datetime.now() - timedelta(minutes=minutes_late)
        ).isoformat(timespec="seconds"),
    }
    fields.update(extra)
    return Reminder(**fields)


def _service(db, *, speak=None, notify=None, publish=None) -> ReminderService:
    remote = MagicMock(enabled=False)
    return ReminderService(
        db,
        speak=speak or (lambda text, sid: "/static/audio/x.wav"),
        notify=notify or AsyncMock(return_value=True),
        publish=publish or (lambda kind, payload: None),
        remote=remote,
    )


# ---------------------------------------------------------------------------
#  Partitioning
# ---------------------------------------------------------------------------
def test_just_due_reminder_fires_normally():
    fresh, stale = partition_due_reminders([_row(0)])
    assert len(fresh) == 1 and not stale


def test_reminder_inside_the_grace_window_still_fires():
    """A minute or two late is still the reminder you asked for."""
    fresh, stale = partition_due_reminders([_row(5)])
    assert len(fresh) == 1 and not stale


def test_reminder_past_the_grace_window_is_stale():
    fresh, stale = partition_due_reminders([_row(60)])
    assert not fresh and len(stale) == 1


def test_weekend_backlog_does_not_stampede():
    """The actual failure: three days offline must not fire everything."""
    due = [_row(m, reminder_id=i) for i, m in enumerate([1, 90, 2000, 4300])]
    fresh, stale = partition_due_reminders(due)
    assert [r.reminder_id for r in fresh] == [0]
    assert [r.reminder_id for r in stale] == [1, 2, 3]


def test_unparseable_timestamp_is_treated_as_fresh():
    """Refusing to fire is a worse failure than firing something stale."""
    fresh, stale = partition_due_reminders([_row(0, remind_at="not a date")])
    assert len(fresh) == 1 and not stale


def test_partition_uses_the_supplied_clock():
    now = datetime(2026, 8, 10, 12, 0, 0)
    due = [
        Reminder(reminder_id=1, text="a", remind_at="2026-08-10T11:55:00"),
        Reminder(reminder_id=2, text="b", remind_at="2026-08-10T09:00:00"),
    ]
    fresh, stale = partition_due_reminders(due, now=now)
    assert [r.reminder_id for r in fresh] == [1]
    assert [r.reminder_id for r in stale] == [2]


# ---------------------------------------------------------------------------
#  Digest
# ---------------------------------------------------------------------------
def test_missed_reminders_are_retired_and_summarised():
    db = get_db()
    two_hours_ago = (datetime.now() - timedelta(hours=2)).isoformat(timespec="seconds")
    ids = [db.add_reminder(f"missed {n}", two_hours_ago) for n in range(3)]
    stale = [
        Reminder(reminder_id=rid, text=f"missed {n}", remind_at=two_hours_ago)
        for n, rid in enumerate(ids)
    ]

    published: list[tuple[str, dict]] = []
    notify = AsyncMock(return_value=True)
    service = _service(
        db, notify=notify, publish=lambda kind, payload: published.append((kind, payload))
    )

    asyncio.run(service.deliver_missed_digest(stale))

    # One event and one Telegram message for the whole backlog, not three.
    assert [kind for kind, _ in published] == ["reminder_digest"]
    assert published[0][1]["count"] == 3
    assert notify.await_count == 1

    # And they are retired as 'missed', not silently claimed as delivered.
    pending = {r.reminder_id for r in db.get_pending_reminders(limit=100)}
    assert not pending & set(ids)


def test_one_utterance_for_the_whole_digest():
    """Three missed reminders must not mean three synthesis passes."""
    calls = []
    service = _service(MagicMock(), speak=lambda text, sid: calls.append(text) or "")

    asyncio.run(service.deliver_missed_digest([_row(200, reminder_id=i) for i in range(3)]))
    assert len(calls) == 1


def test_ancient_reminders_are_dropped_from_the_digest():
    """Last month's reminder is noise, not information."""
    published = []
    notify = AsyncMock(return_value=True)
    db = MagicMock()
    service = _service(db, notify=notify, publish=lambda kind, payload: published.append(kind))

    asyncio.run(service.deliver_missed_digest([_row(60 * 24 * 40, reminder_id=1)]))  # 40 days

    db.mark_reminder_missed.assert_called_once_with(1)
    assert published == []
    notify.assert_not_awaited()
