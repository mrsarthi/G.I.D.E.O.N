"""
G.I.D.E.O.N Composition Root

The one place that decides which concrete services exist and how they are
wired together. Interfaces receive what they need from here instead of
reaching into each other's modules (the web server used to import the
Telegram bot to deliver reminders, and the Telegram bot imported the
orchestrator's internals).

Construction is cheap and does no I/O beyond opening SQLite. start() does the
expensive work — vector store, voice models, background loops, Telegram — so
the web app can be created (and tested) without starting the engine.
"""

from __future__ import annotations

import asyncio

from gideon.brain.agent import Agent
from gideon.brain.approvals import ApprovalQueue
from gideon.brain.llm import PowerGuard, warm_up
from gideon.brain.titles import SessionTitler
from gideon.core.config import QUICK_SESSION_ID
from gideon.core.events import EventBus, bus
from gideon.core.logs import get_logger
from gideon.interfaces.telegram import GideonTelegramDaemon
from gideon.reminders.service import ReminderService
from gideon.storage.db import DatabaseManager, close_db, get_db
from gideon.storage.memory import VectorMemory
from gideon.storage.remote_scheduler import RemoteScheduler, get_remote_scheduler
from gideon.voice.tts import SpeechSynthesizer

log = get_logger("gideon.app")


class Services:
    def __init__(
        self,
        db: DatabaseManager | None = None,
        memory: VectorMemory | None = None,
        speech: SpeechSynthesizer | None = None,
        remote: RemoteScheduler | None = None,
        events: EventBus = bus,
    ) -> None:
        self.db = db or get_db()
        self.memory = memory or VectorMemory()
        self.speech = speech or SpeechSynthesizer()
        self.remote = remote or get_remote_scheduler()
        self.events = events

        self.approvals = ApprovalQueue(self.db, publish=self.events.publish_threadsafe)
        self.agent = Agent(self.db, self.memory, self.approvals)
        self.titler = SessionTitler(self.db, publish=self.events.publish_threadsafe)
        self.power = PowerGuard(publish=self.events.publish_threadsafe)
        self.telegram = GideonTelegramDaemon(agent=self.agent)
        self.reminders = ReminderService(
            self.db,
            speak=self.speech.synthesize,
            notify=self._notify_telegram,
            publish=self.events.publish,
            remote=self.remote,
        )
        self._tasks: list[asyncio.Task] = []

    async def _notify_telegram(self, text: str, chat_id: str | None, audio_url: str) -> bool:
        audio_path = self.speech.path_for(audio_url)
        return await self.telegram.notify_reminder(
            text, chat_id=chat_id, audio_path=str(audio_path) if audio_path else None
        )

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        self.events.bind_loop(asyncio.get_running_loop())
        # The quick-ask window opens straight onto a fixed session; creating it
        # here means the hotkey never has to round-trip through session creation.
        self.db.ensure_session(QUICK_SESSION_ID, "Quick Ask")
        # Opening Chroma and loading ONNX voices take seconds; keep the loop free.
        await asyncio.to_thread(lambda: self.memory.collection)
        await asyncio.to_thread(self.speech.load)

        self._tasks = [
            asyncio.create_task(self.reminders.run_delivery_loop()),
            asyncio.create_task(self.reminders.run_remote_sync_loop()),
            # First check runs immediately: booting on battery must not leave
            # a GPU-resident model in place for even one request.
            asyncio.create_task(self.power.run()),
            asyncio.create_task(warm_up()),
        ]
        await self.telegram.start()
        log.info("G.I.D.E.O.N online")

    async def stop(self) -> None:
        # Telegram first: a hard stop skipped here is what leaves the
        # getUpdates consumer registered and makes the next start hit 409.
        await self.telegram.stop()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks = []
        self.remote.close()
        self.memory.close()
        self.speech.shutdown()
        self.titler.shutdown()
        close_db()
        log.info("G.I.D.E.O.N offline")
