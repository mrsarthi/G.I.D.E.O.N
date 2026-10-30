"""
G.I.D.E.O.N Approval Queue

Privileged tool calls (the shell, launching programs) do not run on the
model's say-so alone when policy requires a human in the loop. The agent
files them here instead; an interface shows the request, and the user's
click — not another model turn — is what executes it.

That last point is the design. The old gate told the model "ask the user to
confirm", but nothing recorded the answer, so the next turn hit the same gate
and the tool could never run. Worse, a model that *could* approve its own
request would be no gate at all against instructions injected by a web page.

Statuses: pending -> running -> approved | failed, or pending -> denied |
expired. The pending -> running transition is a conditional UPDATE, so a
double-click or a second tab cannot execute a command twice.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from gideon.core.config import APPROVAL_TTL_MINUTES
from gideon.core.logs import get_logger
from gideon.storage.db import DatabaseManager
from gideon.storage.models import PendingAction
from gideon.tools.registry import dispatch_tool_call

log = get_logger("gideon.brain.approvals")

# Thread-safe publisher: approvals arrive from FastAPI worker threads.
Publish = Callable[[str, dict[str, Any]], None]


class ApprovalError(Exception):
    """The action cannot be approved or denied in its current state."""


def action_payload(action: PendingAction) -> dict[str, Any]:
    """What interfaces need to render an approval card or its outcome."""
    return {
        "action_id": action.action_id,
        "tool": action.tool_name,
        "args": action.tool_args,
        "summary": action.describe(),
        "status": action.status,
        "result": action.result,
        "created_at": action.created_at,
    }


class ApprovalQueue:
    def __init__(self, db: DatabaseManager, publish: Publish | None = None) -> None:
        self.db = db
        self.publish = publish

    def _expire_stale(self) -> None:
        cutoff = datetime.now() - timedelta(minutes=APPROVAL_TTL_MINUTES)
        expired = self.db.expire_actions(cutoff.isoformat(timespec="seconds"))
        if expired:
            log.info("Expired %d unapproved action(s)", expired)

    def request(
        self,
        session_id: str,
        channel: str,
        tool_name: str,
        args: dict[str, Any],
        trace_id: str | None = None,
    ) -> PendingAction:
        action = self.db.create_action(session_id, channel, tool_name, args, trace_id=trace_id)
        log.info("Approval requested #%s: %s", action.action_id, action.describe())
        return action

    def pending(self, session_id: str | None = None) -> list[PendingAction]:
        self._expire_stale()
        return self.db.list_actions(session_id=session_id, status="pending")

    def recent(self, session_id: str, minutes: int = 30, limit: int = 5) -> list[PendingAction]:
        """Actions from this session the model should know the outcome of."""
        self._expire_stale()
        since = (datetime.now() - timedelta(minutes=minutes)).isoformat(timespec="seconds")
        return self.db.list_actions(session_id=session_id, since=since, limit=limit)

    def _get(self, action_id: int) -> PendingAction:
        self._expire_stale()
        action = self.db.get_action(action_id)
        if action is None:
            raise ApprovalError(f"No action #{action_id}.")
        return action

    def approve(self, action_id: int) -> PendingAction:
        """Execute the action. Raises ApprovalError unless it is still pending."""
        action = self._get(action_id)
        if not self.db.resolve_action(action_id, "running", expect="pending"):
            raise ApprovalError(f"Action #{action_id} is {action.status}, not pending.")

        log.info("Approved #%s: %s", action_id, action.describe())
        try:
            result = dispatch_tool_call(action.tool_name, action.tool_args, channel=action.channel)
            status = "approved"
        except Exception as exc:
            log.exception("Approved action #%s failed", action_id)
            result, status = f"The {action.tool_name} tool failed: {exc}", "failed"

        self.db.resolve_action(action_id, status, result, expect="running")
        # Same bookkeeping the agent writes for tools it runs itself, so the
        # session history shows what was executed and when.
        self.db.insert_message(
            action.session_id,
            "tool_call",
            f"[Tool Call: {action.tool_name}] (approved #{action_id})",
            tool_name=action.tool_name,
            tool_args=json.dumps(action.tool_args),
        )
        self.db.insert_message(action.session_id, "tool", result, tool_name=action.tool_name)

        resolved = self._get(action_id)
        self._announce(resolved)
        return resolved

    def deny(self, action_id: int) -> PendingAction:
        action = self._get(action_id)
        if not self.db.resolve_action(action_id, "denied", expect="pending"):
            raise ApprovalError(f"Action #{action_id} is {action.status}, not pending.")
        log.info("Denied #%s: %s", action_id, action.describe())
        # A denial is the clearest correction there is: record it on the turn
        # that asked (PRD §6.3).
        if action.trace_id:
            try:
                self.db.record_correction(
                    action.trace_id, kind="approval_denied", text=action.describe()
                )
            except Exception as exc:
                log.debug("Could not record denial on trace %s: %s", action.trace_id, exc)
        resolved = self._get(action_id)
        self._announce(resolved)
        return resolved

    def _announce(self, action: PendingAction) -> None:
        if self.publish is not None:
            try:
                self.publish("action_resolved", action_payload(action))
            except Exception as exc:
                log.debug("Could not publish action outcome: %s", exc)
