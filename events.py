"""
G.I.D.E.O.N Event Bus

A single always-on channel the frontend listens to, so G.I.D.E.O.N can speak
without being spoken to first: due reminders, hardware warnings, background
task progress.

Publishers may be async (FastAPI handlers) or plain threads (the TTS pool,
the reminder daemon), so both a coroutine and a thread-safe entry point are
provided.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from config import get_logger

log = get_logger("gideon.events")

# Slow/dead browser tabs must never block the daemons that feed them.
_QUEUE_MAXSIZE = 64


class EventBus:
    """Fan-out pub/sub over asyncio queues, one queue per connected client."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the serving loop so worker threads can publish into it."""
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        self._subscribers.add(queue)
        log.debug("Client subscribed (%d active)", len(self._subscribers))
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)
        log.debug("Client unsubscribed (%d active)", len(self._subscribers))

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def _build(self, event_type: str, payload: dict[str, Any] | None) -> str:
        return json.dumps(
            {
                "type": event_type,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                **(payload or {}),
            }
        )

    def publish(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Publish from inside the event loop."""
        message = self._build(event_type, payload)
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                # A client that cannot keep up loses events rather than
                # stalling the publisher.
                log.warning("Dropping event for a saturated subscriber")

    def publish_threadsafe(
        self, event_type: str, payload: dict[str, Any] | None = None
    ) -> None:
        """Publish from a worker thread that is not running the loop."""
        if self._loop is None or self._loop.is_closed():
            log.debug("Event %s dropped: no bound loop", event_type)
            return
        self._loop.call_soon_threadsafe(self.publish, event_type, payload)


bus = EventBus()
