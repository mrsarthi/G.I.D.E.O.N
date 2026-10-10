"""Reading traces and attaching the user's feedback to them.

Token-gated: traces hold full conversation text.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from gideon.app import Services
from gideon.interfaces.api.deps import get_services, require_api_token

router = APIRouter(dependencies=[Depends(require_api_token)])


class Feedback(BaseModel):
    # +1 good, -1 bad.
    rating: int = Field(..., ge=-1, le=1)
    note: str | None = Field(None, max_length=2000)


@router.get("/api/traces")
def list_traces(limit: int = 50, services: Services = Depends(get_services)):
    return services.db.list_traces(limit=min(max(limit, 1), 500))


@router.get("/api/traces/{trace_id}")
def get_trace(trace_id: str, services: Services = Depends(get_services)):
    trace = services.db.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Unknown trace.")
    return trace


@router.post("/api/traces/{trace_id}/feedback")
def rate_trace(trace_id: str, feedback: Feedback, services: Services = Depends(get_services)):
    if feedback.rating == 0:
        raise HTTPException(status_code=422, detail="Rating must be 1 or -1.")
    if not services.agent.traces.rate(trace_id, feedback.rating, feedback.note):
        raise HTTPException(status_code=404, detail="Unknown trace.")
    return {"status": "recorded"}
