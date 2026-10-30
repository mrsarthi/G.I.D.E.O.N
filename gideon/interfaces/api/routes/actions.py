"""Approving or refusing privileged actions the agent has asked to run.

Token-gated like the terminal: an approval executes a shell command, so any
local process able to POST here unauthenticated would have a shell.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from gideon.app import Services
from gideon.brain.approvals import ApprovalError, action_payload
from gideon.interfaces.api.deps import get_services, require_api_token

router = APIRouter(dependencies=[Depends(require_api_token)])


@router.get("/api/actions")
def list_pending_actions(
    session_id: str | None = None, services: Services = Depends(get_services)
):
    """Actions still waiting for a decision, newest first."""
    return [action_payload(a) for a in services.approvals.pending(session_id)]


@router.post("/api/actions/{action_id}/approve")
def approve_action(action_id: int, services: Services = Depends(get_services)):
    """Run the action now. Blocks until it finishes, like the terminal drawer."""
    try:
        return action_payload(services.approvals.approve(action_id))
    except ApprovalError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/api/actions/{action_id}/deny")
def deny_action(action_id: int, services: Services = Depends(get_services)):
    try:
        return action_payload(services.approvals.deny(action_id))
    except ApprovalError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
