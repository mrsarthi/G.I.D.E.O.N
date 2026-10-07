"""Mirroring reminders to the always-on node.

The point of the mirror is that a reminder fires when this machine is off. The
risk it introduces is a reminder firing twice — once from each side. Delivery
ownership is therefore exclusive and per row, and these tests pin both halves:
the node never loses a reminder to a network blip, and the desktop never
double-sends one the node already owns.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from gideon.reminders.service import ReminderService
from gideon.storage.db import get_db
from gideon.storage.models import Reminder
from gideon.storage.remote_scheduler import RemoteScheduler, wire_payload


def _node(client=None) -> RemoteScheduler:
    """A configured scheduler whose HTTP client is a mock."""
    return RemoteScheduler(url="https://node.example", token="secret", client=client or MagicMock())


def _service(remote, *, db=None, notify=None, published=None) -> ReminderService:
    return ReminderService(
        db or MagicMock(),
        speak=lambda text, sid: "",
        notify=notify or AsyncMock(return_value=True),
        publish=(lambda kind, payload: published.append(kind)) if published is not None
        else (lambda kind, payload: None),
        remote=remote,
    )


def _reminder(reminder_id: int, synced: bool, hours_late: float = 0, **extra) -> Reminder:
    return Reminder(
        reminder_id=reminder_id,
        text=f"r{reminder_id}",
        remind_at=(datetime.now() - timedelta(hours=hours_late)).isoformat(timespec="seconds"),
        remote_synced=synced,
        **extra,
    )


# ---------------------------------------------------------------------------
#  Delivery ownership
# ---------------------------------------------------------------------------
def test_desktop_owns_delivery_when_no_node_is_configured():
    service = _service(RemoteScheduler(url="", token=""))
    assert service.owes_telegram(_reminder(1, synced=False)) is True
    assert service.owes_telegram(_reminder(1, synced=True)) is True


def test_node_owns_delivery_for_synced_rows():
    assert _service(_node()).owes_telegram(_reminder(1, synced=True)) is False


def test_desktop_keeps_unsynced_rows_even_with_a_node():
    """A push that failed must not leave the reminder owned by nobody."""
    assert _service(_node()).owes_telegram(_reminder(1, synced=False)) is True


def test_url_without_token_is_not_a_configured_node():
    """An unauthenticated node would let anyone message your Telegram chat."""
    assert RemoteScheduler(url="https://node.example", token="").enabled is False


def test_synced_reminder_is_not_sent_twice():
    """The desktop shows the HUD but leaves Telegram to the node."""
    remote = _node()
    remote.cancel = MagicMock(return_value=True)
    notify = AsyncMock(return_value=True)

    asyncio.run(_service(remote, notify=notify).deliver(_reminder(7, synced=True)))

    notify.assert_not_awaited()
    # ...and the node's copy is withdrawn so it cannot arrive later.
    remote.cancel.assert_called_once_with(7)


def test_unsynced_reminder_is_still_sent_by_the_desktop():
    notify = AsyncMock(return_value=True)
    reminder = _reminder(8, synced=False, channel="telegram", chat_id="999")

    asyncio.run(_service(_node(), notify=notify).deliver(reminder))

    notify.assert_awaited_once()
    assert notify.await_args.args[1] == "999"


# ---------------------------------------------------------------------------
#  The client is fail-soft
# ---------------------------------------------------------------------------
def test_push_is_a_noop_when_not_configured():
    client = MagicMock()
    remote = RemoteScheduler(url="", token="", client=client)
    assert remote.push(_reminder(1, synced=False)) is False
    client.put.assert_not_called()


def test_push_survives_an_unreachable_node():
    """Losing the network must never lose the row."""
    client = MagicMock()
    client.put.side_effect = OSError("no route to host")
    assert _node(client).push(_reminder(1, synced=False)) is False


def test_push_reports_success_on_201():
    client = MagicMock()
    client.put.return_value = MagicMock(status_code=201)
    assert _node(client).push(_reminder(1, synced=False)) is True


def test_payload_carries_no_conversation_data():
    """Only the reminder itself leaves the machine."""
    payload = wire_payload(
        Reminder(
            reminder_id=3,
            text="dentist",
            remind_at="2026-01-01T09:00:00",
            chat_id="42",
            channel="telegram",
        )
    )
    assert set(payload) == {"reminder_id", "text", "remind_at", "chat_id", "channel"}


def test_cancel_treats_404_as_success():
    """The node not holding it is the outcome we wanted either way."""
    client = MagicMock()
    client.delete.return_value = MagicMock(status_code=404, json=lambda: {"prior": None})
    assert _node(client).cancel(5) is True


def test_withdraw_reports_what_the_node_did_with_it():
    client = MagicMock()
    client.delete.return_value = MagicMock(
        status_code=200, json=lambda: {"status": "cancelled", "prior": "sent"}
    )
    assert _node(client).withdraw(5) == (True, "sent")


def test_withdraw_survives_a_node_with_no_body():
    """An older node still counts as acknowledged; we just learn nothing."""
    client = MagicMock()
    response = MagicMock(status_code=204)
    response.json.side_effect = ValueError("no body")
    client.delete.return_value = response
    assert _node(client).withdraw(5) == (True, None)


# ---------------------------------------------------------------------------
#  The digest must not cry wolf
# ---------------------------------------------------------------------------
def test_reminder_the_node_delivered_is_not_reported_as_missed():
    """The whole feature: it fired on your phone while the PC was off. Coming
    back to 'you missed this' would make the digest untrustworthy."""
    db = MagicMock()
    remote = _node()
    remote.withdraw = MagicMock(return_value=(True, "sent"))
    published: list[str] = []
    notify = AsyncMock(return_value=True)

    service = _service(remote, db=db, notify=notify, published=published)
    asyncio.run(service.deliver_missed_digest([_reminder(1, synced=True, hours_late=3)]))

    db.mark_reminder_completed.assert_called_once_with(1)
    db.mark_reminder_missed.assert_not_called()
    assert published == []
    notify.assert_not_awaited()


def test_reminder_the_node_never_fired_is_still_reported():
    """Node configured but broken must not swallow the reminder silently."""
    db = MagicMock()
    remote = _node()
    remote.withdraw = MagicMock(return_value=(True, "pending"))
    published: list[str] = []

    service = _service(remote, db=db, published=published)
    asyncio.run(service.deliver_missed_digest([_reminder(2, synced=True, hours_late=3)]))

    db.mark_reminder_missed.assert_called_once_with(2)
    assert published == ["reminder_digest"]


def test_unreachable_node_falls_back_to_reporting_it_missed():
    """Not knowing must resolve toward telling you, not toward silence."""
    db = MagicMock()
    remote = _node()
    remote.withdraw = MagicMock(return_value=(False, None))
    published: list[str] = []

    service = _service(remote, db=db, published=published)
    asyncio.run(service.deliver_missed_digest([_reminder(3, synced=True, hours_late=3)]))

    db.mark_reminder_missed.assert_called_once_with(3)
    assert published == ["reminder_digest"]


# ---------------------------------------------------------------------------
#  Retry queue
# ---------------------------------------------------------------------------
def test_sync_marks_accepted_rows():
    remote = _node()
    remote.push = MagicMock(return_value=True)
    db = MagicMock()
    db.get_unsynced_reminders.return_value = [_reminder(1, False), _reminder(2, False)]

    assert remote.sync_pending(db) == 2
    assert db.mark_reminder_synced.call_count == 2


def test_sync_stops_hammering_a_down_node():
    remote = _node()
    remote.push = MagicMock(return_value=False)
    db = MagicMock()
    db.get_unsynced_reminders.return_value = [_reminder(i, False) for i in range(10)]

    assert remote.sync_pending(db) == 0
    assert remote.push.call_count == 1


def test_unsynced_query_ignores_past_due_rows():
    """Asking the node to fire something already late is the digest's job."""
    db = get_db()
    past = db.add_reminder(
        "already late", (datetime.now() - timedelta(hours=1)).isoformat(timespec="seconds")
    )
    future = db.add_reminder(
        "still coming", (datetime.now() + timedelta(hours=1)).isoformat(timespec="seconds")
    )

    ids = {r.reminder_id for r in db.get_unsynced_reminders(limit=200)}
    assert future in ids
    assert past not in ids

    db.mark_reminder_completed(past)
    db.mark_reminder_completed(future)


def test_marking_synced_removes_a_row_from_the_queue():
    db = get_db()
    rid = db.add_reminder(
        "mirror me", (datetime.now() + timedelta(hours=2)).isoformat(timespec="seconds")
    )
    assert rid in {r.reminder_id for r in db.get_unsynced_reminders(limit=200)}

    db.mark_reminder_synced(rid, True)
    assert rid not in {r.reminder_id for r in db.get_unsynced_reminders(limit=200)}

    db.mark_reminder_completed(rid)


def test_due_reminders_expose_sync_state():
    """The daemon cannot decide ownership without this column."""
    db = get_db()
    rid = db.add_reminder(
        "sync probe", (datetime.now() - timedelta(minutes=1)).isoformat(timespec="seconds")
    )
    db.mark_reminder_synced(rid, True)
    row = [r for r in db.get_due_reminders() if r.reminder_id == rid][0]
    assert row.remote_synced is True
    db.mark_reminder_completed(rid)
