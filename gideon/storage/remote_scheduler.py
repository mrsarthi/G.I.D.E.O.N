"""
G.I.D.E.O.N Remote Scheduler Client

Mirrors reminders to a small always-on node (see node/) so they fire even when
this machine is shut down. The node stores nothing but the reminder text, its
due time and the origin chat_id, and it only *sends* on Telegram — it never
polls, because Telegram permits exactly one getUpdates consumer per token and
that consumer is interfaces/telegram.py here.

Delivery ownership is per row, tracked by reminders.remote_synced:

    synced   -> the node fires it on Telegram; this process only shows the HUD
    unsynced -> the node never received it, so this process still sends it

Exactly one side sends any given reminder, so there is nothing to
de-duplicate, and a node outage degrades to local-only delivery rather than to
silence.

Every method here is fail-soft. A reminder that cannot be mirrored is still a
reminder; losing the network must never lose the row.
"""

from __future__ import annotations

import threading
from typing import Any

import httpx

from gideon.core.config import (
    REMOTE_SCHEDULER_TIMEOUT,
    REMOTE_SCHEDULER_TOKEN,
    REMOTE_SCHEDULER_URL,
)
from gideon.core.logs import get_logger
from gideon.storage.models import Reminder

log = get_logger("gideon.remote_scheduler")


def wire_payload(reminder: Reminder) -> dict[str, Any]:
    """Build the wire payload.

    Deliberately minimal. `channel` is carried only so the node can label the
    outgoing message; nothing else about the conversation is transmitted.
    """
    return {
        "reminder_id": int(reminder.reminder_id),
        "text": str(reminder.text),
        "remind_at": str(reminder.remind_at),
        "chat_id": str(reminder.chat_id) if reminder.chat_id else None,
        "channel": reminder.channel or "web",
    }


class RemoteScheduler:
    def __init__(
        self,
        url: str = REMOTE_SCHEDULER_URL,
        token: str = REMOTE_SCHEDULER_TOKEN,
        timeout: float = REMOTE_SCHEDULER_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        # One pooled client. Building one per reminder pays TLS setup each time.
        self._client = client
        self._client_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """Both halves are required: an unauthenticated node would let anyone
        schedule messages into your Telegram chat."""
        return bool(self.url and self.token)

    def _get_client(self) -> httpx.Client:
        with self._client_lock:
            if self._client is None:
                self._client = httpx.Client(
                    base_url=self.url,
                    timeout=self.timeout,
                    headers={"Authorization": f"Bearer {self.token}"},
                )
            return self._client

    def close(self) -> None:
        with self._client_lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

    # ------------------------------------------------------------------
    #  Operations
    # ------------------------------------------------------------------
    def push(self, reminder: Reminder) -> bool:
        """Mirror one reminder to the node. Returns True if the node accepted it.

        Never raises: the caller has already committed the row locally, and a
        failure here only means this process keeps delivery responsibility.
        """
        if not self.enabled:
            return False

        try:
            response = self._get_client().put(
                f"/reminders/{int(reminder.reminder_id)}", json=wire_payload(reminder)
            )
        except Exception as exc:
            log.warning("Remote scheduler unreachable; keeping local delivery: %s", exc)
            return False

        if response.status_code in (200, 201):
            return True

        # 401/403 means the token is wrong — worth shouting about, because every
        # reminder will silently fall back to local-only delivery until fixed.
        if response.status_code in (401, 403):
            log.error(
                "Remote scheduler rejected our token (HTTP %s). Reminders will only "
                "fire while this machine is running until GIDEON_REMOTE_SCHEDULER_TOKEN "
                "matches the node's.",
                response.status_code,
            )
        else:
            log.warning(
                "Remote scheduler refused reminder #%s: HTTP %s %s",
                reminder.reminder_id,
                response.status_code,
                response.text[:200],
            )
        return False

    def withdraw(self, reminder_id: int) -> tuple[bool, str | None]:
        """Withdraw a reminder from the node and learn what became of it.

        Returns (acknowledged, prior_status). `prior_status` is what the node
        had the row as before deleting it — "sent" means the node already
        delivered it while this machine was off, which is the difference
        between "you missed this" and "this was handled".

        Without that answer, every reminder the node delivered on time would
        come back as a false entry in the missed-reminder digest on the next
        boot, making the digest untrustworthy.
        """
        if not self.enabled:
            return False, None
        try:
            response = self._get_client().delete(f"/reminders/{int(reminder_id)}")
        except Exception as exc:
            log.debug("Could not withdraw remote reminder #%s: %s", reminder_id, exc)
            return False, None

        # 404 is success from our side: the node is not holding it either way.
        if response.status_code not in (200, 204, 404):
            return False, None

        prior = None
        try:
            prior = response.json().get("prior")
        except Exception:
            # An older node, or a 204 with no body. The withdrawal still
            # counted; we just cannot tell whether it had fired.
            pass
        return True, prior

    def cancel(self, reminder_id: int) -> bool:
        """Withdraw a reminder the node is holding, ignoring its prior state.

        Used when this process delivered the reminder itself, so the node's
        copy must not arrive later.
        """
        return self.withdraw(reminder_id)[0]

    def health(self) -> dict[str, Any] | None:
        """Probe the node. Returns its JSON body, or None if it is unreachable."""
        if not self.enabled:
            return None
        try:
            response = self._get_client().get("/health")
            if response.status_code == 200:
                return response.json()
            log.debug("Remote scheduler health returned HTTP %s", response.status_code)
        except Exception as exc:
            log.debug("Remote scheduler health check failed: %s", exc)
        return None

    def sync_pending(self, db: Any) -> int:
        """Push every unsynced future reminder. Returns how many were accepted.

        Called on startup and on a retry timer, so a reminder created while the
        node was down still reaches it before it comes due.
        """
        if not self.enabled:
            return 0

        try:
            rows = db.get_unsynced_reminders()
        except Exception:
            log.exception("Could not read unsynced reminders")
            return 0

        synced = 0
        for reminder in rows:
            if self.push(reminder):
                db.mark_reminder_synced(reminder.reminder_id, True)
                synced += 1
            else:
                # The node is down. Stop hammering it for the rest of this
                # pass; the next timer tick retries from the top.
                break

        if synced:
            log.info("Mirrored %d reminder(s) to the remote scheduler", synced)
        return synced


# ---------------------------------------------------------------------------
#  Process-wide instance
# ---------------------------------------------------------------------------
# Tools run without injected dependencies, so set_reminder reaches the
# mirror through this accessor; the composition root uses the same instance.
_instance: RemoteScheduler | None = None
_instance_lock = threading.Lock()


def get_remote_scheduler() -> RemoteScheduler:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = RemoteScheduler()
    return _instance
