"""Conversation sessions and the user profile."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from gideon.app import Services
from gideon.brain.titles import PLACEHOLDER_TITLE
from gideon.core.logs import get_logger
from gideon.interfaces.api.deps import get_services

log = get_logger("gideon.api.sessions")

router = APIRouter()


class SessionCreate(BaseModel):
    name: str | None = None


class ProfileUpdateRequest(BaseModel):
    key: str
    value: str


@router.get("/api/sessions")
def list_sessions(services: Services = Depends(get_services)):
    try:
        return services.db.list_sessions()
    except Exception as exc:
        log.exception("Listing sessions failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/api/sessions")
def create_session(data: SessionCreate, services: Services = Depends(get_services)):
    try:
        # No name means "name it for me": a placeholder until the first
        # exchange, then a generated title (brain/titles.py).
        name = (data.name or "").strip()
        auto_title = not name
        session_id = services.db.create_session(name or PLACEHOLDER_TITLE, auto_title=auto_title)
        name = name or PLACEHOLDER_TITLE
        return {"session_id": session_id, "session_name": name}
    except Exception as exc:
        log.exception("Creating session failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, services: Services = Depends(get_services)):
    try:
        services.db.delete_session(session_id)
        if services.memory.is_open:
            services.memory.delete_session(session_id)
        return {"status": "success", "message": f"Session {session_id} deleted."}
    except Exception as exc:
        log.exception("Deleting session failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/api/sessions/{session_id}/history")
def get_session_history(session_id: str, services: Services = Depends(get_services)):
    try:
        return services.db.get_history(session_id)
    except Exception as exc:
        log.exception("Fetching history failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/api/profile")
def get_user_profile(services: Services = Depends(get_services)):
    try:
        return services.db.get_user_profile()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/api/profile")
def update_user_profile(data: ProfileUpdateRequest, services: Services = Depends(get_services)):
    try:
        services.db.set_user_profile(data.key, data.value)
        return {"status": "success", "profile": services.db.get_user_profile()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
