"""Chat (streamed, spoken) and the always-on event stream."""

from __future__ import annotations

import asyncio
import json
import time
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
        # Approval cards and the trace id, raised mid-turn; flushed into the
        # stream ahead of the next text so the page can attach feedback.
        side_events: list[dict] = []
        trace_ids: list[str] = []

        def on_event(event: dict) -> None:
            if "trace_id" in event:
                trace_ids.append(event["trace_id"])
            side_events.append(event)

        reply: list[str] = []
        started = time.perf_counter()
        first_audio_ms: int | None = None
        audio_count = 0
        try:
            tokens = services.agent.stream(session_id, message, on_event=on_event)
            for event in speak_as_you_go(tokens, services.speech, session_id, data.speak):
                if "text" in event:
                    reply.append(event["text"])
                elif "audio_url" in event:
                    audio_count += 1
                    if first_audio_ms is None:
                        first_audio_ms = round((time.perf_counter() - started) * 1000)
                while side_events:
                    yield _sse(side_events.pop(0))
                yield _sse(event)
            while side_events:
                yield _sse(side_events.pop(0))
            yield _sse({"done": True})
            services.titler.maybe_title(session_id, message, "".join(reply))
            # The agent traced the turn; speech timing is only known here.
            if trace_ids and data.speak:
                services.agent.traces.annotate(
                    trace_ids[0], first_audio_ms=first_audio_ms, audio_clips=audio_count
                )
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
