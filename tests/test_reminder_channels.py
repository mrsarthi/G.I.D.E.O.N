"""Cross-channel reminders.

A reminder must be settable from Telegram, and a fired reminder must reach
Telegram regardless of where it was created — otherwise it only exists in a
browser tab that may not be open.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from gideon.storage.db import get_db


# ---------------------------------------------------------------------------
#  Creating reminders from Telegram
# ---------------------------------------------------------------------------
def test_set_reminder_is_available_over_telegram():
    """Unlike the shell, reminders are safe to expose remotely."""
    from gideon.tools.registry import tools_for_channel

    names = {t["function"]["name"] for t in tools_for_channel("telegram")}
    assert "set_reminder" in names
    assert "list_reminders" in names


def test_reminder_records_telegram_origin():
    from gideon.core.context import turn
    from gideon.tools.reminders import set_reminder

    with turn("telegram", "telegram_7922377249"):
        result = set_reminder("call the dentist", "in 2 hours")

    assert "saved for" in result
    # Rows come back ordered by due time, so select by text rather than position.
    rows = get_db().get_due_reminders(
        (datetime.now() + timedelta(days=400)).isoformat(timespec="seconds")
    )
    row = [r for r in rows if r.text == "call the dentist"][-1]
    assert row.channel == "telegram"
    assert row.chat_id == "7922377249"


def test_reminder_from_web_has_no_chat_id():
    from gideon.core.context import turn
    from gideon.tools.reminders import set_reminder

    with turn("web", "some-uuid-session"):
        set_reminder("desktop reminder", "in 3 hours")

    rows = get_db().get_due_reminders(
        (datetime.now() + timedelta(days=400)).isoformat(timespec="seconds")
    )
    row = [r for r in rows if r.text == "desktop reminder"][-1]
    assert row.channel == "web"
    assert row.chat_id is None


def test_chat_id_is_derived_only_for_telegram():
    from gideon.core.context import get_turn_origin, turn

    with turn("web", "telegram_123"):
        channel, chat_id = get_turn_origin()

    assert (channel, chat_id) == ("web", None)


# ---------------------------------------------------------------------------
#  Delivering fired reminders to Telegram
# ---------------------------------------------------------------------------
def _daemon(allowed=(111, 222), send_voice=False):
    """A real daemon with Telegram's transport replaced."""
    from gideon.interfaces.telegram import GideonTelegramDaemon

    daemon = GideonTelegramDaemon(
        token="FAKE", allowed_users=set(allowed), send_voice=send_voice
    )
    daemon.app = MagicMock()
    daemon.send_message = AsyncMock(return_value=True)
    daemon.send_voice_note = AsyncMock(return_value=True)
    return daemon


def test_reminder_with_origin_chat_goes_to_that_chat():
    daemon = _daemon()
    daemon.broadcast = AsyncMock(return_value=2)

    ok = asyncio.run(daemon.notify_reminder("⏰ take a break", chat_id="7922377249"))

    assert ok is True
    daemon.send_message.assert_awaited_once_with("7922377249", "⏰ take a break")
    daemon.broadcast.assert_not_awaited()


def test_desktop_reminder_broadcasts_to_all_authorised_users():
    """The whole point: set at the desk, delivered to the phone."""
    daemon = _daemon()

    ok = asyncio.run(daemon.notify_reminder("⏰ stand up", chat_id=None))

    assert ok is True
    assert daemon.send_message.await_count == 2


def test_notify_is_a_noop_when_telegram_is_not_running():
    daemon = _daemon()
    daemon.app = None

    assert asyncio.run(daemon.notify_reminder("⏰ nobody home")) is False
    daemon.send_message.assert_not_awaited()


def test_voice_note_only_sent_when_enabled():
    daemon = _daemon(send_voice=False)
    asyncio.run(daemon.notify_reminder("⏰ x", chat_id="1", audio_path="C:/tmp/a.wav"))
    daemon.send_voice_note.assert_not_awaited()

    daemon = _daemon(send_voice=True)
    asyncio.run(daemon.notify_reminder("⏰ x", chat_id="1", audio_path="C:/tmp/a.wav"))
    daemon.send_voice_note.assert_awaited_once_with("1", "C:/tmp/a.wav")


# ---------------------------------------------------------------------------
#  Schema
# ---------------------------------------------------------------------------
def test_due_reminders_expose_origin_fields():
    """The daemon needs channel/chat_id to route delivery."""
    db = get_db()
    db.add_reminder(
        "schema probe",
        (datetime.now() - timedelta(minutes=1)).isoformat(timespec="seconds"),
        channel="telegram",
        chat_id="999",
    )
    rows = db.get_due_reminders()
    probe = [r for r in rows if r.text == "schema probe"][-1]
    assert (probe.channel, probe.chat_id) == ("telegram", "999")
    db.mark_reminder_completed(probe.reminder_id)


def test_legacy_reminders_without_origin_still_load():
    """Rows created before the columns existed default to web/None."""
    db = get_db()
    with db._lock, db.conn:
        db.conn.execute(
            "INSERT INTO reminders (text, remind_at) VALUES (?, ?)",
            ("legacy row", (datetime.now() - timedelta(minutes=1)).isoformat(timespec="seconds")),
        )
    row = [r for r in db.get_due_reminders() if r.text == "legacy row"][-1]
    assert row.channel == "web"
    assert row.chat_id is None
    db.mark_reminder_completed(row.reminder_id)
