"""Chat (streamed, spoken) and the always-on event stream."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from gideon.app import Services
from gideon.core.logs import get_logger
from gideon.interfaces.api.deps import get_services
from gideon.voice.streaming import speak_as_you_go

log = get_logger("gideon.api.chat")

router = APIRouter()

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class MessageRequest(BaseModel):
    session_id: str
    message: str
    # The quick-ask window is silent by design, so synthesising for it burns
    # CPU that the model is competing for and writes .wav files nobody plays.
    # Defaulted True so the HUD is unaffected.
    speak: bool = True


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/api/chat")
def chat(data: MessageRequest, services: Services = Depends(get_services)):
    """Stream a reply as SSE, speaking each sentence as it completes."""
    session_id = data.session_id
    message = data.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="Message content cannot be empty.")
    if not services.db.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Unknown session_id.")

    services.agent.record_user_message(session_id, message)

    def chat_generator() -> Iterator[str]:
        try:
            tokens = services.agent.stream(session_id, message)
            for event in speak_as_you_go(tokens, services.speech, session_id, data.speak):
                yield _sse(event)
            yield _sse({"done": True})
        except Exception as exc:
            # The response has already started, so an HTTPException here would
            # never reach the client. Send an explicit error event instead.
            log.exception("Chat stream failed")
            yield _sse({"error": str(exc)})
            yield _sse({"done": True})

    return StreamingResponse(
        chat_generator(), media_type="text/event-stream", headers=_SSE_HEADERS
    )


@router.get("/api/events")
async def event_stream(services: Services = Depends(get_services)):
    """Always-on channel for unprompted messages (reminders, alerts)."""
    queue = services.events.subscribe()

    async def generator():
        try:
            yield ": connected\n\n"
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=20)
                    yield f"data: {message}\n\n"
                except TimeoutError:
                    # Comment frame keeps proxies and browsers from timing out.
                    yield ": keepalive\n\n"
        finally:
            services.events.unsubscribe(queue)

    return StreamingResponse(generator(), media_type="text/event-stream", headers=_SSE_HEADERS)
