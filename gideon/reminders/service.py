"""
G.I.D.E.O.N Reminder Delivery

Polls SQLite for due reminders and delivers them to every channel that can
receive them: the HUD (event bus) always, Telegram when this process owns the
row. Two problems the naive "poll and fire" loop got wrong are handled here:

* Lateness. Booting after a weekend used to fire the whole backlog at once,
  each with its own speech. Past REMINDER_MAX_LATENESS_MINUTES a reminder is
  retired as 'missed' and reported in one digest instead.
* Uptime. When the always-on node holds a reminder (remote_synced), the node
  sends the Telegram copy — including while this machine is off — and this
  process only shows the HUD, so nothing is delivered twice.

Every collaborator (speech, Telegram, the bus, the node) is injected, so the
delivery rules are testable without FastAPI, a voice model or a network.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any

from gideon.core.config import (
    REMINDER_DIGEST_MAX_AGE_HOURS,
    REMINDER_MAX_LATENESS_MINUTES,
    REMINDER_POLL_SECONDS,
    REMOTE_SCHEDULER_RETRY_SECONDS,
)
from gideon.core.logs import get_logger
from gideon.storage.models import Reminder
from gideon.storage.remote_scheduler import RemoteScheduler

log = get_logger("gideon.reminders")

# (text, session_id) -> served audio URL, or "" when speech is unavailable.
Speak = Callable[[str, str], str]
# (text, chat_id, audio_url) -> delivered?
Notify = Callable[[str, str | None, str], Awaitable[bool]]
# (event_type, payload)
Publish = Callable[[str, dict[str, Any]], None]


def partition_due_reminders(
    due: list[Reminder], now: datetime | None = None
) -> tuple[list[Reminder], list[Reminder]]:
    """Split due reminders into ones to fire and ones that are merely late.

    A reminder more than REMINDER_MAX_LATENESS_MINUTES overdue came due while
    this machine was asleep or shut down. Firing those normally means coming
    back from a weekend to a stampede — every stale alert spoken in turn, the
    9am Friday one shouting on Monday evening. Past the window they are
    retired and summarised instead.

    Rows with an unparseable remind_at are treated as fresh: refusing to fire
    is a worse failure than firing something slightly stale.
    """
    now = now or datetime.now()
    grace = timedelta(minutes=REMINDER_MAX_LATENESS_MINUTES)
    fresh: list[Reminder] = []
    stale: list[Reminder] = []
    for reminder in due:
        scheduled = reminder.due
        if scheduled is None or now - scheduled <= grace:
            fresh.append(reminder)
        else:
            stale.append(reminder)
    return fresh, stale


class ReminderService:
    def __init__(
        self,
        db: Any,
        *,
        speak: Speak,
        notify: Notify,
        publish: Publish,
        remote: RemoteScheduler,
    ) -> None:
        self.db = db
        self.speak = speak
        self.notify = notify
        self.publish = publish
        self.remote = remote

    def owes_telegram(self, reminder: Reminder) -> bool:
        """Whether this process is responsible for the Telegram copy.

        When the remote scheduler holds a reminder it fires that copy itself —
        that is the point of mirroring, since it runs when this machine does
        not. Sending here too would double every reminder, so ownership is
        exclusive and follows the per-row remote_synced flag.
        """
        if not self.remote.enabled:
            return True
        return not reminder.remote_synced

    # ------------------------------------------------------------------
    #  Delivery
    # ------------------------------------------------------------------
    async def deliver(self, reminder: Reminder) -> None:
        """Fire one on-time reminder to the HUD and, if we own it, Telegram."""
        await asyncio.to_thread(self.db.mark_reminder_completed, reminder.reminder_id)

        honorific = random.choice(["boss", "sir"])
        spoken = f"Reminder {honorific}: {reminder.text}"
        audio_url = await asyncio.to_thread(self.speak, spoken, "reminder")

        # 1. Browser HUD. Always ours — the remote node cannot reach it.
        self.publish(
            "reminder",
            {
                "reminder_id": reminder.reminder_id,
                "text": reminder.text,
                "scheduled_for": reminder.remind_at,
                "message": spoken,
                "audio_url": audio_url,
                "origin": reminder.channel,
            },
        )

        # 2. Telegram, so it reaches you away from the desk. Prefers the
        #    originating chat, else every authorised user.
        delivered = "owned by node"
        if self.owes_telegram(reminder):
            delivered = "not sent"
            try:
                if await self.notify(f"⏰ {spoken}", reminder.chat_id, audio_url):
                    delivered = "sent"
            except Exception:
                log.exception("Telegram reminder delivery failed")
        else:
            # The node has its own copy queued; withdraw it now that the
            # desktop has handled this one, so it does not arrive again later.
            await asyncio.to_thread(self.remote.cancel, reminder.reminder_id)

        log.info(
            "Fired reminder #%s (origin=%s, telegram=%s)",
            reminder.reminder_id,
            reminder.channel,
            delivered,
        )

    async def deliver_missed_digest(self, stale: list[Reminder]) -> None:
        """Retire overdue reminders and report them as a single summary."""
        cutoff = datetime.now() - timedelta(hours=REMINDER_DIGEST_MAX_AGE_HOURS)
        reportable: list[Reminder] = []
        handled_remotely = 0

        for reminder in stale:
            # Rows the node owns were very likely delivered on time while this
            # machine was off — that is the entire point of mirroring them.
            # Ask the node before calling them missed, or the digest cries
            # wolf about every reminder the system actually got right.
            if not self.owes_telegram(reminder):
                acknowledged, prior = await asyncio.to_thread(
                    self.remote.withdraw, reminder.reminder_id
                )
                if acknowledged and prior == "sent":
                    await asyncio.to_thread(
                        self.db.mark_reminder_completed, reminder.reminder_id
                    )
                    handled_remotely += 1
                    continue

            await asyncio.to_thread(self.db.mark_reminder_missed, reminder.reminder_id)
            due = reminder.due
            if due is None or due >= cutoff:
                reportable.append(reminder)

        log.info(
            "Retired %d overdue reminder(s): %d delivered by the node, %d missed, "
            "%d recent enough to report",
            len(stale),
            handled_remotely,
            len(stale) - handled_remotely,
            len(reportable),
        )
        if not reportable:
            return

        lines = []
        for reminder in reportable:
            due = reminder.due
            stamp = due.strftime("%a %d %b, %I:%M %p") if due else str(reminder.remind_at)
            lines.append(f"• [{stamp}] {reminder.text}")

        count = len(reportable)
        headline = f"{count} reminder{'s' if count != 1 else ''} came due while you were offline"
        body = f"{headline}:\n" + "\n".join(lines)

        # One utterance for the whole digest, not one per missed reminder.
        audio_url = await asyncio.to_thread(self.speak, headline, "reminder")

        self.publish(
            "reminder_digest",
            {
                "count": count,
                "message": headline,
                "audio_url": audio_url,
                "items": [
                    {
                        "reminder_id": r.reminder_id,
                        "text": r.text,
                        "scheduled_for": r.remind_at,
                    }
                    for r in reportable
                ],
            },
        )

        # Digests are always ours to send: the node fires individual reminders
        # on time, so anything in here is by definition something nobody
        # delivered. No audio: the headline alone is not the message.
        try:
            await self.notify(f"⏰ {body}", None, "")
        except Exception:
            log.exception("Telegram digest delivery failed")

    async def check_once(self) -> None:
        due = await asyncio.to_thread(self.db.get_due_reminders)
        if not due:
            return
        fresh, stale = partition_due_reminders(due)
        for reminder in fresh:
            await self.deliver(reminder)
        if stale:
            await self.deliver_missed_digest(stale)

    # ------------------------------------------------------------------
    #  Background loops
    # ------------------------------------------------------------------
    async def run_delivery_loop(self) -> None:
        """Fire due reminders to every channel that can receive them.

        Delivery is deliberately unconditional on origin: a reminder set at
        the desktop still needs to reach the phone when no browser is open,
        and one set on Telegram should also surface in the HUD if it is up.
        """
        log.info(
            "Reminder daemon started (%ds poll, %dmin grace)",
            REMINDER_POLL_SECONDS,
            REMINDER_MAX_LATENESS_MINUTES,
        )
        while True:
            try:
                await asyncio.sleep(REMINDER_POLL_SECONDS)
                await self.check_once()
            except asyncio.CancelledError:
                log.info("Reminder daemon stopped")
                raise
            except Exception:
                log.exception("Reminder daemon iteration failed")

    async def run_remote_sync_loop(self) -> None:
        """Keep the always-on node's copy of upcoming reminders current.

        Runs only when a node is configured. A reminder created while the node
        was unreachable stays flagged unsynced and is retried here, so a
        transient outage costs nothing.
        """
        if not self.remote.enabled:
            log.info(
                "Remote scheduler not configured; reminders fire only while this "
                "machine is running"
            )
            return

        node = await asyncio.to_thread(self.remote.health)
        log.info("Remote scheduler %s", "reachable" if node else "unreachable (will retry)")
        while True:
            try:
                await asyncio.to_thread(self.remote.sync_pending, self.db)
            except asyncio.CancelledError:
                log.info("Remote sync daemon stopped")
                raise
            except Exception:
                log.exception("Remote sync iteration failed")
            await asyncio.sleep(REMOTE_SCHEDULER_RETRY_SECONDS)
